import numpy as np
from opt.utils import detect_error, extract_item_list, clean_llm_output


def ndcg(target_rank, N=None):
    if target_rank < 1:
        return 0.0
    if N is not None and target_rank > N:
        return 0.0
    return 1.0 / np.log2(target_rank + 1)



# reward.py  ──────────────────────────────────────────────────────────
from opt.utils import extract_ranking_and_explanations, explanation_similarity, build_local_meta,  parse_freeform_ranking

class Reward:
    def __init__(self, config, request_model):
        """
        item_meta : dict[int] -> {title:str, desc:str|None}
        """
        self.conf    = config
        self.request = request_model
        self.alpha   = config.get("alpha", 0.8)  # weight for NDCG
        self.beta    = config.get("beta",  0.2)  # weight for explanation sim

    def _score_explanations(self, tuples,raw_input):
        local_meta = build_local_meta(raw_input)
        sims = []
        for idx, expl in tuples[:5]:
            if idx in local_meta:
                meta = local_meta[idx]
                sims.append(explanation_similarity(expl, meta["title"], meta["desc"]))
        return np.mean(sims) if sims else 0.0

    
    def calculate_reward(self, prompt, sample_data):
        total = 0.0
        print(f"\n=== Calculating reward for prompt ===")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        debug_info = []  # Store debug information
        
        for i, d in enumerate(sample_data):
            # Initialize debug entry for this sample
            debug_entry = {
                "sample_index": i,
                "input": d['input'],
                "raw_response": "",
                "cleaned_response": "",
                "parsed": None,
                "status": "pending"
            }
            
            raw_resp = self.request.request(d["input"], prompt)
            cleaned_resp = clean_llm_output(raw_resp)
            
            # Update debug entry with response info
            debug_entry["raw_response"] = raw_resp
            debug_entry["cleaned_response"] = cleaned_resp
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Input: {d['input'][:100]}...")
            print(f"Raw response: {raw_resp[:200]}...")
            print(f"Cleaned response: {cleaned_resp[:200]}...")
            
            parsed = extract_ranking_and_explanations(cleaned_resp)
            debug_entry["parsed"] = parsed
            print(f"Parsed items: {len(parsed)}")
            
            # Fallback parsing if primary fails
            if not parsed or len(parsed) != 20:
                parsed_fallback = parse_freeform_ranking(cleaned_resp)  # Use cleaned_resp here
                
                if parsed_fallback and len(parsed_fallback) == 20:
                    parsed = parsed_fallback
                    debug_entry["parsed"] = parsed
                    debug_entry["parsing_method"] = "fallback"
            
            # Validate parsing
            if not parsed or len(parsed) != 20:
                debug_entry["status"] = "skipped"
                debug_entry["reason"] = f"Invalid parsing ({len(parsed) if parsed else 0} items)"
                debug_info.append(debug_entry)
                continue

            # Calculate ranking accuracy
            ranks = [idx for idx, _ in parsed]
            try:
                target_rank = ranks.index(int(d["target_index"])) + 1
            except ValueError:
                target_rank = len(ranks) + 1
            # acc_score = ndcg(target_rank)
            acc_score = ndcg(target_rank, N=self.conf["candidate_size"])

            # Calculate explanation quality
            expl_score = self._score_explanations(parsed, d["input"])

            # Update debug entry
            debug_entry.update({
                "status": "processed",
                "target_rank": target_rank,
                "acc_score": acc_score,
                "expl_score": expl_score
            })
            debug_info.append(debug_entry)
            
            # Accumulate weighted reward
            total += self.alpha * acc_score + self.beta * expl_score
        
        # Print debug info
        print("\n=== REWARD CALCULATION DEBUG ===")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Samples processed: {len([d for d in debug_info if d['status'] == 'processed'])}")
        print(f"Samples skipped: {len([d for d in debug_info if d['status'] == 'skipped'])}")
        
        if debug_info:
            first_skipped = next((d for d in debug_info if d['status'] == 'skipped'), None)
            if first_skipped:
                print("\nFirst skipped sample:")
                print(f"Reason: {first_skipped.get('reason', '')}")
                print(f"Input snippet: {first_skipped.get('input', '')[:200]}...")
                print(f"Response snippet: {first_skipped.get('cleaned_response', '')[:200]}...")
        
        return total
