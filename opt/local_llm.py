from __future__ import annotations
from typing import List
import os, re, torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import warnings
from transformers.utils import logging as transformers_logging
from transformers import LogitsProcessor, LogitsProcessorList


# Option A: filter out *that specific* warning message
warnings.filterwarnings(
    "ignore",
    message=r"Setting `pad_token_id` to `eos_token_id`"
)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    After max_thinking_tokens are generated, inject '</think>' to end the thinking block.
    """

    def __init__(self, tokenizer, max_thinking_tokens: int = 100):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.count = 0
        self.stopped = False

    def __call__(self, input_ids, scores):
        if self.stopped:
            return scores

        self.count = input_ids.size(1)
        if self.count >= self.max_thinking_tokens:
            # Force '</think>' as next token
            forced = torch.full_like(scores, -float("inf"))
            forced[:, self.think_end_id] = 0
            self.stopped = True
            return forced

        return scores

def extract_qwen_final_output(tokenizer, output_ids, think_end_token_id=151668):
    try:
        idx = len(output_ids) - output_ids[::-1].index(think_end_token_id)
    except ValueError:
        idx = 0
    return tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")



# ─────────────────── helpers for memory caps ───────────────────────────────
def _parse_mem_string(s: str) -> dict[int, str]:
    """
    Accept either "0:29GB,1:45GB" *or* "29GB,45GB" and return {gpu_idx: "XGB"}
    """
    out: dict[int, str] = {}
    if ":" in s:                          # explicit mapping
        for part in s.split(","):
            idx, cap = part.split(":")
            out[int(idx.strip())] = cap.strip().upper()
    else:                                 # positional list
        for idx, cap in enumerate(p.strip().upper() for p in s.split(",")):
            out[idx] = cap
    return out


def _auto_max_memory(reserve_gb: int = 2) -> dict[int, str]:
    """
    Compute {gpu_idx: '<n>GB'} from CUDA queries, keeping `reserve_gb` free
    on every card for activations / kernels.
    """
    mem = {}
    for i in range(torch.cuda.device_count()):
        total     = torch.cuda.get_device_properties(i).total_memory // 2**30
        mem[i]    = f"{max(1, total - reserve_gb)}GB"
    return mem


# ───────────────────────── Local LLM wrapper ───────────────────────────────
class LocalLLM:
    """
    Behaviour-compatible replacement for the previous class but powered by
    `transformers.pipeline`.  Call pattern stays identical:

        >>> llm = LocalLLM(max_new_tokens=64)
        >>> llm.chat("1+1?", "You are a pirate!")
        'Arrr! ##2##'

    """
    def __init__(
        self,
        model_id: str,
        torch_dtype: str = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 4,
        **kwargs
    ):
        # Detect backend
        self.is_qwen3 = "qwen3" in model_id.lower()
        self.is_llama = "llama" in model_id.lower()

        # Smart default dtype
        if torch_dtype is None:
            torch_dtype = "auto" if self.is_qwen3 else "auto"  # Llama works well with auto too

        # Load model and tokenizer
        if torch_dtype is None:
            torch_dtype = torch.float16  # Quadro RTX 8000: prefer fp16 over 'auto'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            # torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size


    
    # ───────────────────── private helpers ────────────────────────────────
    def _truncate(self, msg_list):
        """
        Ensure the *encoded* length ≤ ctx_len - max_new_tokens
        (Llama 3 uses special tokens for roles; the tokenizer handles those.)
        """
        enc = self.tokenizer.apply_chat_template(msg_list, add_generation_prompt=True,
                                                 return_tensors="pt")
        excess = enc.size(1) - (self.ctx_len - self.max_new_tokens)
        if excess <= 0:
            return msg_list                                 # nothing to cut

        # naive but safe: chop earliest user content
        trimmed = msg_list.copy()
        while excess > 0 and len(trimmed) > 1:              # keep at least 1 msg
            trimmed.pop(1)                                  # drop first user msg
            enc = self.tokenizer.apply_chat_template(
                trimmed, add_generation_prompt=True,
                return_tensors="pt"
            )
            excess = enc.size(1) - (self.ctx_len - self.max_new_tokens)
        return trimmed

    def _generate(self, msgs_batch: List[List[dict]]) -> List[str]:
        # 1) render chat-lists → single prompt strings
        rendered = [
            self.tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.is_qwen3
            )
            for m in msgs_batch
        ]

        # 2) prepare logits processor to limit thinking for Qwen3
        from transformers import LogitsProcessorList
        proc = LogitsProcessorList()
        if self.is_qwen3:
            proc.append(ThinkingTokenBudgetProcessor(self.tokenizer, max_thinking_tokens=2048))

        # 3) call the HF pipeline's generate via its underlying model
        # Note: Using pipeline, so we need to call model.generate directly
        model_inputs = self.tokenizer(rendered, return_tensors="pt", padding=True).to(self.model.device)
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            logits_processor=proc  # <-- apply our thinking-limiter
        )

        # 4) decode and strip thinking content if needed
        results = []
        for idx, out_ids in enumerate(generated):
            new_tokens = out_ids[len(model_inputs["input_ids"][idx]):].tolist()
            if self.is_qwen3:
                text = extract_qwen_final_output(self.tokenizer, new_tokens)
            else:
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append(text)
        
        return results

    def chat(self, user_prompt: str, system_prompt: str = None) -> str:
        # Build messages as list of dicts (for Qwen3 and Llama3)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Qwen3: enable_thinking, Llama: no extra arg
        kwargs = {}
        if self.is_qwen3:
            kwargs["enable_thinking"] = True

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        if self.is_qwen3:
            # Strip thinking content (token 151668 == </think>)
            try:
                idx = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                idx = 0
            return self.tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
        else:
            # Llama and other LLMs: just decode
            return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def chat_batch(self, user_prompt_list, system_prompt: str = None):
        return [self.chat(up, system_prompt) for up in user_prompt_list]

    
