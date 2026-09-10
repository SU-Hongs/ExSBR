# metrics.py  ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

class Metric:
    def __init__(self, rank_list, conf):
        self.rank_list = rank_list          # list[int] of 1‑based ranks
        self.conf      = conf               # contains 'candidate_size'
        # expl_sim_list will be injected by Eval before run()
        self.len_list      = getattr(self, "len_list",      [])
        self.uniq_ratio_ls = getattr(self, "uniq_ratio_ls", [])
        self.ppl_list      = getattr(self, "ppl_list",      [])

    # ── basic helpers ────────────────────────────────────────────────
    def ndcg(self, N):
        return np.mean([
            0 if r > N else 1 / np.log2(r + 1) for r in self.rank_list
        ])

    def hit(self, N):
        return np.mean([
            0 if r > N else 1 for r in self.rank_list
        ])

    def mean_ap(self, N):
        return np.mean([
            0 if r > N else 1 / r for r in self.rank_list
        ])

    # ── public method ────────────────────────────────────────────────
    def run(self):
        res = pd.DataFrame({"KPI@K": ["NDCG", "HIT", "MAP"]})

        topk = [1, 5, 10] if self.conf["candidate_size"] == 10 else [1, 5, 10, 20]
        for k in topk:
            res[k] = np.array([self.ndcg(k), self.hit(k), self.mean_ap(k)])

        # count how many targets were ranked inside the candidate list
        valid = sum(r <= self.conf["candidate_size"] for r in self.rank_list)
        res["#valid_data"] = np.array([valid, 0, 0])

        # ── NEW: aggregate explanation similarity (ExplSim) ──────────
        if hasattr(self, "expl_sim_list"):
            expl_avg = np.mean(self.expl_sim_list)
        else:
            expl_avg = np.nan
        res["ExplSim"] = np.array([expl_avg, np.nan, np.nan])
        res["AvgLen"]            = np.array([np.mean(self.len_list),       np.nan, np.nan])
        res["UniqueTokenRatio"]  = np.array([np.mean(self.uniq_ratio_ls),  np.nan, np.nan])
        res["Perplexity"]        = np.array([np.mean(self.ppl_list),       np.nan, np.nan])

        return res
