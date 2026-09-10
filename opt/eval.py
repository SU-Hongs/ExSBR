from tqdm import tqdm
from opt.metrics import Metric
from opt.request import Request
from opt.utils import extract_item_list
from opt.utils import clean_llm_output
from opt.utils import (
    extract_ranking_and_explanations, 
    build_local_meta,  # <-- new parser
    explanation_similarity,
    unique_token_ratio,          # <-- new import
    perplexity,
    parse_freeform_ranking             # <-- new metric
)
import math
import numpy as np
import re, ast

def extract_ranking_block(text: str) -> str:
    m = re.search(r'<\s*ranking\s*>.*?<\s*/\s*ranking\s*>',
                  text, flags=re.DOTALL|re.IGNORECASE)
    return m.group(0).strip() if m else text.strip()

class Eval():
    def __init__(self, config, data, text_table):
        self.conf = config
        self.requset = Request(config)
        self.data = data
        self.text_table = text_table
        self.error_list = []
        self.target_rank_list = []
        
    
    def run(self, prompt):
        self.normal_eval(prompt)
        return None, self.target_rank_list, self.error_list
    
    def record_error(self, data, response):
        tmp = {}
        tmp['response'] = response
        tmp['target'] = data['target']
        tmp['input'] = data['input']
        tmp['target_index'] = data['target_index']
        
        return tmp

    # def normal_eval(self, prompt):
    #     for data in tqdm(self.data):
    #         for _ in range(3):
    #             response = self.requset.request(user=data['input'], system=prompt)
    #             result_list = extract_item_list(response, data['target'])
    #             if not result_list:
    #                 continue
    #             elif ((int(result_list[-1])) < self.conf['candidate_size']+1) and (int(result_list[-1]))>0:
    #                 self.target_rank_list.append(int(result_list[-1]))
    #                 break
    #         # self.text_table.add_data(data['input'], data['target'], response)
    #         if self.text_table is not None:
    #             self.text_table.add_data(data['input'], data['target'], response)

    #         if (not result_list) or (int(result_list[-1]) >= (self.conf['candidate_size']+1)):
    #             error = self.record_error(data, response)
    #             self.error_list.append(error)
    #             self.target_rank_list.append(self.conf['candidate_size']+1)
    
    def normal_eval_noexp(self, prompt):
        # batch‑size comes from command line flag --bs (default 4)
        bs = self.conf.get('bs', 4)
        total = len(self.data)

        for start in tqdm(range(0, total, bs)):
            batch          = self.data[start : start + bs]
            input_list     = [x['input']  for x in batch]
            target_list    = [x['target'] for x in batch]

            # one model call for the whole batch
            responses = self.requset.batch_request(input_list, system=prompt)

            for data_entry, target, response in zip(batch, target_list, responses):
                result_list = extract_item_list(response, target)

                if result_list and 0 < int(result_list[-1]) < self.conf['candidate_size'] + 1:
                    self.target_rank_list.append(int(result_list[-1]))
                else:
                    self.target_rank_list.append(self.conf['candidate_size'] + 1)
                    error = self.record_error(data_entry, response)
                    self.error_list.append(error)

                if self.text_table is not None:
                    self.text_table.add_data(data_entry['input'], target, response)
    def normal_eval(self, prompt):
        """
        • Sends batched queries to the LLM.
        • Records target‑item rank (for NDCG / HIT / MAP).
        • Computes semantic quality of explanations (ExplSim).
        """
        bs = self.conf.get("bs", 4)
        total = len(self.data)

        for start in tqdm(range(0, total, bs), desc="EVAL"):
            batch = self.data[start : start + bs]
            inputs = [item["input"] for item in batch]

            # single LLM call for the whole batch
            responses = self.requset.batch_request(inputs, system=prompt)

            for data_entry, response in zip(batch, responses):
                response = clean_llm_output(response)
                response = response.replace("\\n", "\n")
                
                # Then proceed with parsing
                rank_expl = extract_ranking_and_explanations(response, data_entry["input"])
                parsing_method = "primary"
                # ── 1. parse ranking + explanations from the LLM output ──
                # First try to parse with special tags
                # if response.strip().startswith("[") and "generated_text" in response:
                #     try:
                #         import ast
                #         response = ast.literal_eval(response)[0]["generated_text"]
                #     except:
                #         pass
                # response = response.replace("\\n", "\n")
                # rank_expl = extract_ranking_and_explanations(response)
                
                # If tag-based parsing fails, try freeform parsing
                if not rank_expl or len(rank_expl) != 20:
                    error = self.record_error(data_entry, response)
                    error['parsed_count'] = len(rank_expl) if rank_expl else 0
                    error['parsed_sample'] = str(rank_expl[:3]) if rank_expl else ""
                    self.error_list.append(error)
                    rank_expl_fallback = parse_freeform_ranking(response)
                    
                    # Validate fallback parsing
                    if rank_expl_fallback and len(rank_expl_fallback) == 20:
                        rank_expl = rank_expl_fallback
                        parsing_method = "fallback"
                    else:
                        parsing_method = "failed"
                else:
                    parsing_method = "tag-based"

                # ── 2. target‑item rank (accuracy metric) ──
                if rank_expl and len(rank_expl) == 20:
                    ranked_ids = [idx for idx, _ in rank_expl]
                    try:
                        tgt_rank = ranked_ids.index(data_entry["target_index"]) + 1
                    except ValueError:
                        tgt_rank = self.conf["candidate_size"] + 1
                else:
                    tgt_rank = self.conf["candidate_size"] + 1
                self.target_rank_list.append(tgt_rank)

                
                # ── 4. validation and error logging ──
                error_entry = None
                if not rank_expl:
                    error_entry = self.record_error(data_entry, response)
                    error_entry["error_reason"] = "No items parsed"
                elif len(rank_expl) != 20:
                    error_entry = self.record_error(data_entry, response)
                    error_entry["error_reason"] = f"Expected 20 items, got {len(rank_expl)}"
                    error_entry["parsed_items"] = [idx for idx, _ in rank_expl]
                elif tgt_rank > self.conf["candidate_size"]:
                    error_entry = self.record_error(data_entry, response)
                    error_entry["error_reason"] = "Target item not found in top 20"
                
                if error_entry:
                    error_entry["parsing_method"] = parsing_method
                    self.error_list.append(error_entry)
                
                # ── 5. optional text‑table logging ──
                if self.text_table is not None:
                    block = extract_ranking_block(response)
                    # valid if target was ranked within candidate_size
                    is_valid = (tgt_rank <= self.conf["candidate_size"])
                    self.text_table.add_data(
                        data_entry["input"],
                        data_entry["target"],
                        block,
                        is_valid
                    )
