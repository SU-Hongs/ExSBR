import random
import wandb
import json
import random

from tqdm import tqdm
from opt.config import init_config
from opt.request import Request
from opt.reward import Reward
from opt.improve import Improve
from opt.select import Select
from opt.utils import LogTable
import re

def ensure_output_instructions(prompt: str) -> str:
    """Ensure prompt contains required output instructions"""
    required_phrases = [
        "<ranking>",
        "</ranking>",
        "exactly 20 lines",
        "item-index",
        "explanation"
    ]
    
    for phrase in required_phrases:
        if phrase not in prompt:
            prompt += f"\n\nRemember: {phrase.replace('<', '').replace('>', '')} is required in your output!"
    
    return prompt
def ensure_ranking_wrapper(prompt: str) -> str:
    """
    Guarantees that the instruction to wrap output in <ranking>…</ranking>
    appears exactly once in the *preamble* (i.e. before any <START>…<END>).
    """

    # 1) If it already has <ranking>, do nothing.
    if re.search(r'<\s*ranking\s*>', prompt, re.IGNORECASE):
        return prompt

    # 2) Split off any <START>…</END> blocks so we don't inject inside them.
    parts = re.split(r'(<START>.*?</END>)', prompt, flags=re.DOTALL)
    preamble, rest = parts[0], ''.join(parts[1:])

    # 3) Inject our wrapper reminder at the end of the preamble
    preamble = preamble.rstrip() + (
        "\n\nWrap your FINAL OUTPUT between <ranking> and </ranking> tags\n"
    )

    # 4) Re-assemble and return
    return preamble + rest




def generate_argmax_prompt(beam_candidate, val_data, reward_model, result_table):
    sample_data = val_data
    reward_list = [0] * len(beam_candidate)
    tag_presence = [False] * len(beam_candidate)  # Track if tags are present
    
    for index, prompt in enumerate(beam_candidate):
        # Validate tag presence
        has_tags = "<ranking>" in prompt and "</ranking>" in prompt
        tag_presence[index] = has_tags
        
        # Calculate reward
        reward = reward_model.calculate_reward(prompt, sample_data)
        reward_list[index] = reward

        # Strip out the original scaffold so rewards.csv only shows the candidate body
        scaffold = reward_model.conf.get('initial_prompt', '')
        if scaffold and prompt.startswith(scaffold):
            short_prompt = prompt[len(scaffold):].strip()
        else:
            short_prompt = prompt

        # Log only the candidate-specific part
        result_table.add_data(short_prompt, reward, has_tags)
    
    # Prioritize prompts with tags
    valid_prompts = [(r, i) for i, (r, has_tag) in enumerate(zip(reward_list, tag_presence)) if has_tag]
    
    if valid_prompts:
        # Select highest reward from prompts with tags
        max_reward, prompt_index = max(valid_prompts)
    else:
        # Fallback to highest reward even without tags
        prompt_index = reward_list.index(max(reward_list))
    
    return beam_candidate[prompt_index]


if __name__ == '__main__':
    initial_prompt = """You are a session‑based recommendation engine with built‑in explainability.
INPUT  
• A JSON object containing:  
  - “Current session interactions”: a list of items the user has just interacted with.  
  - “Candidate set”: exactly 20 items (each with title and tags). The input order (1–20) defines your **baseline initial ranking**.
TASK  
1. Detect the key themes or genres in the current session interactions.  
2. Infer the user’s preferences from those themes.  
3. Treat the candidate set’s input order as your baseline initial ranking **and** generate a one‑sentence explanation for each item linking it to the inferred preferences.  
4. Use an LLM‑based explanation model to produce a **new ranking** of the same 20 items, again giving each item a one‑sentence justification.  
5. **Act as an independent recommendation judge:** without regard to which list came first, compare the two ranked lists **solely** on which best predicts the user’s likely next interaction—i.e. which list places the most contextually appropriate items earlier based on the session context and inferred preferences. Discard the other and return only the superior ranking.
OUTPUT  
Return exactly 20 lines wrapped in `<ranking>…</ranking>` tags, each line in the form:  
<ranking>  
1. <index> – <item-title> – <explanation>  
2. <index> – <item-title> – <explanation>  
…  
20. <index> – <item-title> – <explanation>  
</ranking>
"""
   
    inferring_reasons = (
        "You are an expert *prompt engineer* for session‑based recommenders.\n\n"
        "1) **Original prompt**:\n"
        "```\n$prompt$\n```\n\n"
        "2) **Session + Candidate set**:\n"
        "```\n$input$\n```\n\n"
        "3) **LM’s raw output**:\n"
        "```\n$response$\n```\n\n"
        "4) **Ground‑truth item** “$target_title$” (ID $target_index$) was placed at position $target_pos$ (1=top).\n"
        "Please give **$N_r$ concise bullet points** explaining *why* this prompt\n"
        "failed to guide the LM to rank the true item at the top (e.g. ambiguous\n"
        "instructions, missing constraints, formatting confusion)."
    )
    refining_prompts = (
    "I'm trying to write a zero-shot recommender prompt that generalizes well across diverse user sessions.\n"
    "My current prompt is: \"$prompt$\"\n"
    "However, it performs poorly on the following example: $error_case$\n\n"
    "Revise the prompt to correct this issue **without overfitting to session-specific content**. Keep it flexible so that it can adapt to different themes inferred from the session.\n\n"
    "Wrap your FINAL OUTPUT between <ranking> and </ranking> tags\n"
    "<START>\n"
    "…write an improved, generalizable prompt…\n"
    "<END>"
)



    augumenting_prompts = augumenting_prompts = (
    "Generate a variation of the following prompt while:\n"
    "- Keeping its semantic meaning\n"
    "- Preserving the `<ranking>…</ranking>` instructions exactly\n\n"
    "Input: $refined_prompt$\n"
    "Output: (your improved prompt goes here)\n\n"
    "Wrap your FINAL OUTPUT between <ranking> and </ranking> tags"
)

    conf = init_config()
    conf['initial_prompt'] = initial_prompt
    conf['inferring_reasons'] = inferring_reasons
    conf['refining_prompts'] = refining_prompts
    conf['augumenting_prompts'] = augumenting_prompts

    opt_request = Request(conf)
    
        # === logging===
    if conf['backend'] == 'openai':
        openai_key = conf['openai_api_key']
    else:
        openai_key = None

    opt_request = Request(conf)

    # always use local CSV logging
    text_table   = LogTable(
        columns=["Input", "Prompt", "Reason", "Improved prompt", "Augmented prompt"],
        save_dir="logs",
        fname_prefix="texts",
    )
    reward_table = LogTable(
        columns=["Prompt", "Reward", "Has_Tags"],
        save_dir="logs",
        fname_prefix="rewards",
    )

    print("parameter initialization is complete")

    with open(f"./Dataset/{conf['dataset']}/Text/train_{conf['train_num']}.json", 'r') as json_file:
        train_data = json.load(json_file)
    for d in train_data:
        d['target_index'] += 1
    with open(f"./Dataset/{conf['dataset']}/Text/valid.json", 'r') as json_file:
        val_data = json.load(json_file)
    for d in val_data:
        d['target_index'] += 1

    beam_candidate = []
    prompt_candidate = []
    random.seed(conf['seed'])

    opt_reward = Reward(conf, opt_request)
    opt_improve = Improve(inferring_reasons, refining_prompts, augumenting_prompts, train_data, conf, opt_request)
    opt_select = Select(train_data, conf, opt_reward)

    print("==============")
    print("The apo algorithm is running...")
    print("==============")
    beam_candidate.append(initial_prompt)
    pbar = tqdm(range(conf['E_2']))
    
    for i in pbar:
        pbar.set_description(f"Epoch {i+1}/{conf['E_2']}")
        prompt_candidate = []
        for prompt in beam_candidate:
            prompt = ensure_output_instructions(prompt)
            raw_candidates = opt_improve.run(prompt, text_table)
            full_candidates = []
            for body in raw_candidates:
                # 1) reconstruct the full prompt by prefixing the original instructions
                full = (
                    initial_prompt
                    + "\n\n"            # separate the scaffold from the body
                    + body              # your model’s proposed “improved prompt”
                )
                # 2) (re-)ensure the <ranking> wrapper is in place
                full = ensure_ranking_wrapper(full)
                full_candidates.append(full)
            prompt_candidate.extend(full_candidates)
        
        print(f"\n[DEBUG] After prompt generation, prompt_candidate has {len(prompt_candidate)} candidates")
        if len(prompt_candidate) < 3:
            print(f"[DEBUG] prompt_candidate content: {prompt_candidate}")
        # beam_candidate = opt_select.run(prompt_candidate)
        if len(prompt_candidate) >= conf['N_o']:
        # we have enough candidates to select a full beam
            beam_candidate = opt_select.run(prompt_candidate)
        else:
            # not enough—just carry forward whatever we got
            beam_candidate = prompt_candidate
        
    pbar.close()
    if not beam_candidate:
        print("[WARNING] beam_candidate is empty, falling back to initial prompt")
        beam_candidate = [initial_prompt]
    # ── DEBUG: print first 3 prompts to verify wrapper is present ──
    print("\n=== Sample beam candidates after tuning ===")
    for idx, p in enumerate(beam_candidate[:3], start=1):
        body = body = p[len(initial_prompt):].lstrip()      # drop the scaffold
        print(f"\n--- Candidate {idx} ---\n{body}\n")
    print("============================================\n")
    # Argmax prompt
    new_prompt = generate_argmax_prompt(beam_candidate, val_data, opt_reward, reward_table)
    text_table.save()
    reward_table.save()
    
    print("Optimize finished")
