import random
from opt.utils import detect_error, extract_edit_prompt, extract_ranking_and_explanations, build_local_meta
import ast, re
def clean_llm_output(raw: str) -> str:
    # 1) If it’s a repr of a list of dicts, grab the “generated_text”
    if raw.strip().startswith("[") and "generated_text" in raw:
        try:
            lst = ast.literal_eval(raw)
            raw = lst[0].get("generated_text", raw)
        except Exception:
            pass
    # 2) Drop any OpenAI / header tokens
    raw = re.sub(r"<\|[^|]+\|>", "", raw)
    # 3) Un-escape literal "\n"
    raw = raw.replace("\\n", "\n")
    return raw.strip()


class Improve():
    def __init__(self,
                inferring_reasons, 
                refining_prompts, 
                augumenting_prompts, 
                train_data,
                config,
                request_model):
        self.inferring_reasons = inferring_reasons
        self.refining_prompts = refining_prompts
        self.augumenting_prompts = augumenting_prompts
        self.train_data = train_data
        self.config = config
        self.request = request_model
        self.used_data = []
    
    # def evaluate_collect_error(self, prompt, data):
    #     errors_list = []
    #     for val in data:
    #         response = self.request.request(val['input'], prompt)
    #         if not detect_error(response, val['target']):
    #             error = {}
    #             error['input'] = val['input']
    #             error['output'] = response
    #             errors_list.append(error)
    
    #     return errors_list

    def evaluate_collect_error(self, prompt, data_batch):
        inputs   = [v['input']  for v in data_batch]
        targets  = [v['target'] for v in data_batch]

        responses = self.request.batch_request(inputs, system=prompt)

        errors_list = []
        for v, resp, tgt in zip(data_batch, responses, targets):
            cleaned = clean_llm_output(resp)
            # if not detect_error(cleaned, tgt):
            #     errors_list.append({'input': v['input'],
            #                         'output': cleaned})
            if not detect_error(cleaned, tgt):
                # parse out the ranking list
                # parse ranking and find where the true item fell
                parsed = extract_ranking_and_explanations(cleaned)
                ids    = [idx for idx,_ in parsed]
                pos    = (ids.index(v['target_index']) + 1) if v['target_index'] in ids else None

                # look up the actual title from the original block
                from opt.utils import build_local_meta
                meta  = build_local_meta(v['input'])
                title = meta.get(v['target_index'], {}).get('title', f"item {v['target_index']}")

                errors_list.append({
                    'input':        v['input'],
                    'response':     cleaned,
                    'target_index': v['target_index'],
                    'target_title': title,
                    'target_pos':   pos
                })
        # print(f"[DEBUG] evaluate_collect_error: Called with {len(data_batch)} samples")
        return errors_list
    # def evaluate_collect_error(self, prompt, data_batch):
    #     """
    #     For each sample:
    #      • If target_index == -1: skip (out of top‑20 → LM is “innocent”).
    #      • Else parse the LLM’s new <ranking>…</ranking>.
    #      • Compute new_pos (1‑based) vs. baseline_pos (also 1‑based).
    #      • Flag an error only if parsing failed, target missing, or new_pos > baseline_pos.
    #     """
    #     inputs   = [v['input']   for v in data_batch]
    #     responses = self.request.batch_request(inputs, system=prompt)

    #     errors_list = []
    #     for v, raw_resp in zip(data_batch, responses):
    #         # Skip examples where the true item wasn’t even in the top‑20 to begin with
    #         if v['target_index'] < 0:
    #             continue

    #         cleaned = clean_llm_output(raw_resp)

    #         # 1) parse out (item_idx, explanation) tuples
    #         parsed = extract_ranking_and_explanations(cleaned)
    #         ids = [idx for idx, _ in parsed] if parsed else []

    #         # 2) find the new position (1‑based) of the true item
    #         baseline_pos = v['target_index'] + 1
    #         new_pos = ids.index(baseline_pos) + 1 if baseline_pos in ids else None

    #         # 3) flag only if parse failed, missing, or dropped lower
    #         if not parsed or new_pos is None or new_pos > baseline_pos:
    #             meta  = build_local_meta(v['input'])
    #             title = meta.get(baseline_pos, {}).get('title',
    #                                                    f"item {baseline_pos}")

    #             errors_list.append({
    #                 'input':        v['input'],
    #                 'response':     cleaned,
    #                 'target_index': baseline_pos,
    #                 'target_title': title,
    #                 'target_pos':   new_pos
    #             })

    #     return errors_list


    def generate_similar_prompt(self, prompt_list):
        similar_prompts = []
        for prompt in prompt_list:
            if "<ranking>" not in prompt:
                prompt = "Wrap your FINAL OUTPUT between <ranking> and </ranking> tags\n\n" + prompt
            tmp = self.augumenting_prompts
            content = tmp.replace("$refined_prompt$", prompt)
            for i in range(self.config['addition_sample']):
                # 1) call the LLM with content as the user prompt (no empty system)
                raw = self.request.request(content)
                # 2) clean it
                cleaned = clean_llm_output(raw)
                similar_prompts.append(cleaned)
    
        return similar_prompts

    def run(self, prompt, table=None):
        # print(f"[DEBUG] Improve.run called, prompt starts: {prompt[:80]} ...")
        
        candidate_prompts = []
        batch_data = random.sample(self.train_data, self.config['N_t'])
        # print(f"[DEBUG] Improve.run batch_data size: {len(batch_data)}")
        self.used_data += batch_data
        errors_list = self.evaluate_collect_error(prompt, batch_data) 
        # print(f"[DEBUG] Improve.run errors_list size: {len(errors_list)}")
        try:
            errors_group = random.sample(errors_list, self.config['N_e'])
        except:
            errors_group = errors_list
        inferring_reasons = self.inferring_reasons.replace("$prompt$", prompt).replace("$N_r$", str(self.config['N_r'])) 
        refining_prompts = self.refining_prompts.replace("$prompt$", prompt)
        
        for error in errors_group:
            # Inferring reasons for errors
            content = inferring_reasons \
                .replace("$prompt$",       prompt) \
                .replace("$input$",        error["input"]) \
                .replace("$response$",     error["response"]) \
                .replace("$target_index$", str(error["target_index"])) \
                .replace("$target_title$",  error["target_title"]) \
                .replace("$target_pos$",    str(error["target_pos"])) \
                .replace("$N_r$",           str(self.config["N_r"]))
            raw_grad = self.request.request(content)
            gradient = clean_llm_output(raw_grad)

            # 2) Refining the prompt using those reasons
            content = refining_prompts \
                        .replace("$prompt$", prompt) \
                        .replace("$error_case$", error['input']) \
                        .replace("$reasons$", gradient)
            raw_edit = self.request.request(content)
            clean_edit = clean_llm_output(raw_edit)
            edit_prompt_list = extract_edit_prompt(clean_edit)
            # tmp_prompt = inferring_reasons
            # content = tmp_prompt.replace("$error_case$", error['input']) 
            # gradient = self.request.request(user=content, system='')

            # # Refining prompts with reasons
            # tmp_prompt = refining_prompts
            # tmp_prompt = tmp_prompt.replace("$error_case$", error['input']) 
            # content = tmp_prompt.replace("$reasons$", gradient)
            # edit_prompt = self.request.request(user=content, system='')
            # edit_prompt_list = extract_edit_prompt(edit_prompt)

            # Augumenting prompts
            similar_prompts = self.generate_similar_prompt(edit_prompt_list)

            # Merge candidate prompts
            candidate_prompts.extend(edit_prompt_list)
            candidate_prompts.extend(similar_prompts)
            
            # add data into wandb Text Table [input, prompt, reason, improved prompt, augumented prompt]
            if table is not None:
                # for new_index, body in enumerate(edit_prompt_list):
                #     full_improved = prompt + "\n\n" + body
                #     full_improved = clean_llm_output(full_improved)

                #     # each improved prompt produced `addition_sample` augmentations
                #     for mc_index in range(self.config['addition_sample']):
                #         raw_aug = similar_prompts[
                #             new_index * self.config['addition_sample'] + mc_index
                #         ]
                #         full_aug = prompt + "\n\n" + raw_aug
                #         full_aug = clean_llm_output(full_aug)

                #         table.add_data(
                #             error['input'],     # user session
                #             prompt,             # old prompt
                #             gradient,           # why it failed
                #             full_improved,      # improved prompt
                #             full_aug            # augmented prompt
                #         )
                for new_index, body in enumerate(edit_prompt_list):
                    improved = clean_llm_output(body)
                    for mc_index in range(self.config['addition_sample']):
                        aug_body = similar_prompts[new_index*self.config['addition_sample'] + mc_index]
                        augmented = clean_llm_output(aug_body)

                        table.add_data(
                            error['input'],   # session + candidates
                            prompt,           # old prompt
                            gradient,         # self‑reflected reasons
                            improved,         # *only* the LLM’s new prompt text
                            augmented         # *only* the LLM’s variation
                        )

            # if table is not None :
            #     for new_index, new_prompt in enumerate(edit_prompt_list):
            #         for mc_index in range(self.config['addition_sample']):
            #             table.add_data(error['input'], prompt, gradient, new_prompt, similar_prompts[new_index * self.config['addition_sample'] + mc_index])
        # Randomly sampled #num successor candidates per parent prompt
        try:
            sample_candidate_prompts = random.sample(candidate_prompts, self.config['num_candidates'])
        except:
            sample_candidate_prompts = candidate_prompts
        print(f"[DEBUG] Improve.run returning {len(candidate_prompts)} prompts")
        return sample_candidate_prompts
    
    def get_used_data(self):
        return self.used_data