#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined post-hoc evaluation for (A) explainability (AvgLen & UTR only) and (B) ranking accuracy.

Inputs
  - --csv: Path to eval_texts_*.csv (columns: ["Input","Target","Response","Valid"])
  - --raw_json: (optional) Path to raw test_seed_*.json with fields:
        {"input": "...", "target": "...", "target_index": <0-based int>}.
        Used to align rows robustly and fetch the canonical target index.
        If omitted, we recover the target index by exact canonical title equality.
  - --topk: K for explainability's top-K aggregation (default=5; must be <= candidate_size)
  - --candidate_size: candidate set size in the input (default=20)

Outputs (under --out_dir):
  - explain_summary.json/csv   (AvgLen & UTR only; both Top-1 and mean over Top-K)
  - ranking_summary.json/csv   (HR/MRR/NDCG @ {1,5,10,20} intersected with candidate_size)
Also prints compact summaries to stdout.

NOTE: The explainability aggregation follows the user's original pipeline:
  - parse ranking+explanations from Response using opt.utils helpers
  - compute per-row AvgLen and UTR for Top-1 and mean over Top-K
  - aggregate via (population) mean; rows that fail to parse contribute neutral values
"""

import argparse, json, math, re, html, hashlib, unicodedata
from pathlib import Path
from statistics import mean, pstdev
import pandas as pd

# ── user-provided helpers (must be available on PYTHONPATH) ────────────────
from opt.utils import (
    build_local_meta,
    unique_token_ratio,
    extract_ranking_and_explanations,
    parse_freeform_ranking,
    clean_llm_output,
    normalize_indices,
)

# ── safe stats ─────────────────────────────────────────────────────────────
def _is_nan(v):
    return isinstance(v, float) and math.isnan(v)

def safe_mean(x):
    xs = [v for v in x if v is not None and not _is_nan(v)]
    return float(mean(xs)) if xs else float("nan")

def safe_std(x):
    xs = [v for v in x if v is not None and not _is_nan(v)]
    return float(pstdev(xs)) if len(xs) > 1 else float("nan")

# ── Explainability (AvgLen & UTR only) ─────────────────────────────────────
def explainability_summary(csv_path: str, topk=5, candidate_size=20):
    df = pd.read_csv(csv_path)
    if not {"Input","Response"}.issubset(df.columns):
        raise ValueError("CSV must have columns: Input, Response (and usually Target, Valid)")

    parsed_flags   = []
    top1_len_list  = []
    top1_utr_list  = []
    mean_len_list  = []
    mean_utr_list  = []

    for _, r in df.iterrows():
        ipt = str(r["Input"])
        rsp = clean_llm_output(str(r["Response"]))

        # attempt structured parse; fallback to freeform rank parsing
        pairs = extract_ranking_and_explanations(rsp, raw_input=ipt)
        if not pairs:
            pairs = parse_freeform_ranking(rsp)

        # normalize to within candidate_size (if indices present)
        if pairs:
            try:
                pairs = normalize_indices(pairs, candidate_size)
            except Exception:
                pass

        parsed_flags.append(1 if pairs else 0)
        if not pairs:
            # neutral/NaN contributions for aggregation
            top1_len_list.append(0)
            top1_utr_list.append(float("nan"))
            mean_len_list.append(float("nan"))
            mean_utr_list.append(float("nan"))
            continue

        head = pairs[:min(topk, len(pairs))]

        lens, utrs = [], []
        for _, expl in head:
            lens.append(len(expl.split()))
            try:
                utrs.append(unique_token_ratio(expl))  # (#unique tokens / #all tokens)
            except Exception:
                utrs.append(float("nan"))

        top1_len_list.append(lens[0] if lens else 0)
        top1_utr_list.append(utrs[0] if utrs else float("nan"))
        mean_len_list.append(safe_mean(lens))
        mean_utr_list.append(safe_mean(utrs))

    n = len(df)
    n_parsed = sum(parsed_flags)
    parsed_rate = float(n_parsed) / n if n else 0.0

    summary = {
        "rows_total": n,
        "rows_parsed": n_parsed,
        "parsed_rate": parsed_rate,
        "topK": int(topk),
        "candidate_size": int(candidate_size),

        # ONLY these four explainability metrics are reported (as requested)
        "top1_len_mean":      safe_mean(top1_len_list),
        "top1_utr_mean":      safe_mean(top1_utr_list),
        "mean_len_topK_mean": safe_mean(mean_len_list),
        "mean_utr_topK_mean": safe_mean(mean_utr_list),
    }
    return summary

# ── Canonicalization for titles (deterministic, review-safe) ───────────────
def _to_ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def canon_text(s: str, keep_newlines=False) -> str:
    if not isinstance(s, str): return ""
    s = html.unescape(s).replace("\\n", "\n")
    s = re.sub(r"<\|[^|]+\|>", "", s)    # strip <|TOKENS|>
    if not keep_newlines:
        s = re.sub(r"\s+", " ", s).strip()
    else:
        s = s.strip()
    return s

def canon_title(s: str) -> str:
    s = canon_text(s)
    s = _to_ascii(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def key_for_align(inp: str, tgt: str) -> str:
    core = canon_title(inp) + "||" + canon_title(tgt)
    return hashlib.sha1(core.encode("utf-8")).hexdigest()

# ── Parse candidate meta from Input ────────────────────────────────────────
def parse_candidate_meta(raw_input: str):
    raw_input = canon_text(raw_input, keep_newlines=True)
    try:
        block = raw_input.split("Candidate set:")[1]
    except IndexError:
        return {}
    pat = re.compile(
        r'\s+(\d+)\.\s*"([^"]+)"\s*\n\s*Tags:\s*([^\n]*?)\s*(?:\n|$)',
        flags=re.MULTILINE,
    )
    meta = {}
    for m in pat.finditer(block):
        idx   = int(m.group(1))
        title = m.group(2).strip()
        meta[idx] = {"title": title, "norm": canon_title(title)}
    return meta

# ── Ranking blocks & parsing ───────────────────────────────────────────────
def extract_all_blocks(response: str):
    resp = canon_text(response, keep_newlines=True)
    blocks = []
    for m in re.finditer(r'(?:^|\n)(.*?(final output|re\-ranking|original ranking|ranking|output).*)?\n*\s*<\s*ranking\s*>(.*?)<\s*/\s*ranking\s*>',
                         resp, flags=re.DOTALL|re.IGNORECASE):
        header = (m.group(1) or "").strip().lower()
        block  = m.group(3).strip()
        blocks.append((header, block))
    if not blocks:
        for m in re.finditer(r'<\s*ranking\s*>(.*?)<\s*/\s*ranking\s*>',
                             resp, flags=re.DOTALL|re.IGNORECASE):
            blocks.append(("", m.group(1).strip()))
    if not blocks and resp:
        blocks = [("", resp)]
    return blocks

def split_rank_lines(block: str):
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) <= 1:
        tmp = re.sub(r'(?<!^)\s+(?=(\d+[.)]\s))', '\n', block)
        lines = [ln for ln in tmp.splitlines() if ln.strip()]
    return lines

def parse_block_core(block: str):
    p_line = re.compile(r'^\s*\d+[.)]\s*"?(\d+)"?', re.IGNORECASE)
    p_any  = re.compile(r'(^|\s)\d+[.)]\s*"?(\d+)"?', re.IGNORECASE)
    p_tq   = re.compile(r'^\s*\d+[.)]\s*"([^"]+)"\s*[-–—:]\s*', re.IGNORECASE)
    p_tl   = re.compile(r'^\s*\d+[.)]\s*([^"-].*?)\s*[-–—:]\s*', re.IGNORECASE)

    lines = split_rank_lines(block)
    idxs, titles = [], []

    for ln in lines:
        m = p_line.match(ln)
        if m:
            idxs.append(int(m.group(1)))
        else:
            mt = p_tq.match(ln) or p_tl.match(ln)
            if mt:
                titles.append(mt.group(1).strip())

    if len(idxs) <= 1:
        idxs = [int(m.group(2)) for m in p_any.finditer(block)]

    return idxs, titles, len(lines)

def normalize_1based(idxs, candidate_size):
    if not idxs: return idxs, False
    need_shift = (0 in idxs) or ((max(idxs) == candidate_size - 1) and (candidate_size not in idxs))
    if need_shift:
        return [i + 1 for i in idxs], True
    return idxs, False

def finalize_indices(idxs_1b, titles, meta, candidate_size):
    seen, out, dups, oob = set(), [], 0, 0
    for i in idxs_1b:
        if 1 <= i <= candidate_size:
            if i not in seen:
                out.append(i); seen.add(i)
            else:
                dups += 1
        else:
            oob += 1
    title2idx = {v["norm"]: i for i, v in meta.items()}
    mapped = 0
    for t in titles:
        nt = canon_title(t)
        if nt in title2idx:
            idx = title2idx[nt]
            if 1 <= idx <= candidate_size and idx not in seen:
                out.append(idx); seen.add(idx); mapped += 1
    return out, {"duplicates_removed": dups, "out_of_range": oob, "title_only_mapped": mapped}

def parse_block(block, meta, candidate_size):
    raw_idxs, titles, n_lines = parse_block_core(block)
    idxs_1b, shifted = normalize_1based(raw_idxs, candidate_size)
    indices, stats = finalize_indices(idxs_1b, titles, meta, candidate_size)
    return indices, {"lines": n_lines, "raw_found": len(raw_idxs), "shifted": shifted, **stats}

def choose_best_block(blocks, meta, candidate_size):
    scored = []
    for pos, (hdr, blk) in enumerate(blocks):
        idxs, diag = parse_block(blk, meta, candidate_size)
        score = (len(idxs), 1 if "final" in hdr else 0, pos)
        scored.append((score, blk, idxs, diag))
    if not scored:
        return "", [], {"reason": "no_blocks"}
    scored.sort(key=lambda x: x[0])
    _, best_blk, best_idxs, best_diag = scored[-1]
    return best_blk, best_idxs, best_diag

# ── Ranking metrics ────────────────────────────────────────────────────────
def hr_at_k(rank, k):   return 1.0 if (rank is not None and rank <= k) else 0.0
def mrr_at_k(rank, k):  return (1.0 / rank) if (rank is not None and rank <= k) else 0.0
def ndcg_at_k(rank, k): return (1.0 / math.log2(rank + 1)) if (rank is not None and rank <= k) else 0.0

def ranking_summary(csv_path: str, raw_json_path: str|None, candidate_size=20, topk_list=(1,5,10,20)):
    dfm = pd.read_csv(csv_path)
    need = {"Input","Target","Response","Valid"}
    if need - set(dfm.columns):
        raise ValueError(f"CSV must contain columns {need}")

    if raw_json_path:
        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        dfb = pd.DataFrame({
            "key": [key_for_align(r.get("input",""), r.get("target","")) for r in raw],
            "input": [r.get("input","") for r in raw],
            "target": [r.get("target","") for r in raw],
            "target_index": [r.get("target_index",-1) for r in raw],
        }).drop_duplicates("key", keep="first")

        dfmk = pd.DataFrame({
            "key": [key_for_align(r["Input"], r["Target"]) for _, r in dfm.iterrows()],
            "Input": dfm["Input"],
            "Target": dfm["Target"],
            "Response": dfm["Response"],
        }).drop_duplicates("key", keep="first")

        common = set(dfb["key"]) & set(dfmk["key"])
        print(f"# rows in baseline json  : {len(dfb)}")
        print(f"# rows in model csv      : {len(dfmk)}")
        print(f"# matched keys (unique)  : {len(common)}")

        df = dfb.merge(dfmk, on="key", how="inner")
        print(f"Scoring on matched unique rows: {len(df)}")
    else:
        df = dfm.copy()
        df["input"] = df["Input"]
        df["target"] = df["Target"]
        df["target_index"] = None

    rows = []
    for i, r in df.iterrows():
        meta = parse_candidate_meta(r["input"])

        # determine 1-based true index
        if raw_json_path and isinstance(r["target_index"], int) and r["target_index"] >= 0:
            true_idx_1 = r["target_index"] + 1
            target_in_cand = (1 <= true_idx_1 <= candidate_size)
        else:
            true_idx_1 = None
            nt = canon_title(r["target"])
            for idx, v in meta.items():
                if v["norm"] == nt:
                    true_idx_1 = idx
                    break
            target_in_cand = (true_idx_1 is not None)

        blocks = extract_all_blocks(r["Response"])
        _, indices, _ = choose_best_block(blocks, meta, candidate_size)

        tgt_rank = None
        if target_in_cand and indices:
            try:
                tgt_rank = indices.index(true_idx_1) + 1
            except ValueError:
                tgt_rank = None

        rec = {
            "row_id": i,
            "target_in_candidates": bool(target_in_cand),
            "parsed_items": len(indices),
            "target_rank": tgt_rank if tgt_rank is not None else float("inf"),
        }
        for k in [k for k in topk_list if k <= candidate_size]:
            rec[f"HR@{k}"]   = hr_at_k(tgt_rank, k)
            rec[f"MRR@{k}"]  = mrr_at_k(tgt_rank, k)
            rec[f"NDCG@{k}"] = ndcg_at_k(tgt_rank, k)
        rows.append(rec)

    per_row = pd.DataFrame(rows)
    agg = {"#rows": len(per_row),
           "#target_in_candidates": int(per_row["target_in_candidates"].sum())}
    for k in [k for k in topk_list if k <= candidate_size]:
        agg[f"HR@{k}"]   = float(per_row[f"HR@{k}"].mean())
        agg[f"MRR@{k}"]  = float(per_row[f"MRR@{k}"].mean())
        agg[f"NDCG@{k}"] = float(per_row[f"NDCG@{k}"].mean())

    return agg, per_row

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Combined post-hoc evaluation (Explainability: AvgLen & UTR; Ranking: HR/MRR/NDCG)")
    ap.add_argument("--csv", required=True, help="Path to eval_texts_*.csv")
    ap.add_argument("--raw_json", default=None, help="(Optional) Path to raw test_seed_*.json for ground-truth target_index alignment")
    ap.add_argument("--out_dir", default="logs_posthoc", help="Output directory")
    ap.add_argument("--topk", type=int, default=5, help="K for explainability top-K metrics")
    ap.add_argument("--candidate_size", type=int, default=20, help="Candidate size used in inputs")
    ap.add_argument("--topk_list", default="1,5,10,20", help="Comma-separated K list for ranking metrics")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    topk_list = tuple(int(x) for x in str(args.topk_list).split(",") if str(x).strip())

    # ---- Explainability (AvgLen & UTR only) ----
    ex_sum = explainability_summary(args.csv, topk=args.topk, candidate_size=args.candidate_size)

    print("\n=== Explainability (aggregate; AvgLen & UTR only) ===")
    print(f"file                : {args.csv}")
    print(f"rows_total/parsed   : {ex_sum['rows_total']} / {ex_sum['rows_parsed']}  (parsed_rate={ex_sum['parsed_rate']:.3f})")
    print(f"topK / cand_size    : {ex_sum['topK']} / {ex_sum['candidate_size']}")
    print(f"top1_len_mean       : {ex_sum['top1_len_mean']:.2f}")
    print(f"top1_utr_mean       : {ex_sum['top1_utr_mean']:.4f}")
    print(f"mean_len_topK_mean  : {ex_sum['mean_len_topK_mean']:.2f}")
    print(f"mean_utr_topK_mean  : {ex_sum['mean_utr_topK_mean']:.4f}")

    with open(out_dir / "explain_summary.json", "w", encoding="utf-8") as f:
        json.dump(ex_sum, f, ensure_ascii=False, indent=2)
    pd.DataFrame([ex_sum]).to_csv(out_dir / "explain_summary.csv", index=False)

    # ---- Ranking (HR/MRR/NDCG) ----
    rk_sum, rk_rows = ranking_summary(args.csv, args.raw_json, candidate_size=args.candidate_size, topk_list=topk_list)

    print("\n=== Ranking accuracy (aggregate) ===")
    for k, v in rk_sum.items():
        if isinstance(v, float):
            print(f"{k:24s}: {v:.6f}")
        else:
            print(f"{k:24s}: {v}")

    with open(out_dir / "ranking_summary.json", "w", encoding="utf-8") as f:
        json.dump(rk_sum, f, ensure_ascii=False, indent=2)
    pd.DataFrame([rk_sum]).to_csv(out_dir / "ranking_summary.csv", index=False)
    rk_rows.to_csv(out_dir / "ranking_per_row.csv", index=False)

    print(f"\n[✓] Saved explainability  → {out_dir / 'explain_summary.json'} & .csv")
    print(f"[✓] Saved ranking         → {out_dir / 'ranking_summary.json'} & .csv")
    print(f"[✓] Saved per-row ranking → {out_dir / 'ranking_per_row.csv'}")

if __name__ == "__main__":
    main()
