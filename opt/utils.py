import random
import time
import openai
import re
import json
import numpy as np
import ast

import evaluate, nltk, math
from sentence_transformers import SentenceTransformer, util as st_util
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from datetime import datetime

def make_run_log(dir_: str = "logs",
                 prefix: str = "bad_samples") -> str:
    """
    Create <dir_>/<prefix>_YYYYMMDD_HHMMSS.jsonl and return its path.
    Caller stores this in conf['bad_log_fp'] and every blocked request
    simply appends one JSON line to it.
    """
    Path(dir_).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(dir_) / f"{prefix}_{ts}.jsonl"
    # create empty file
    path.touch(exist_ok=False)
    return str(path)

_tok  = nltk.tokenize.TreebankWordTokenizer()            # fast word-level
_umodel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_ppl   = evaluate.load("perplexity", model_id="gpt2")    # 117 M, tiny

def unique_token_ratio(text: str) -> float:
    tokens = _tok.tokenize(text.lower())
    return 0.0 if not tokens else len(set(tokens)) / len(tokens)

def semantic_similarity(sent_a: str, sent_b: str) -> float:
    ea = _umodel.encode(sent_a, convert_to_tensor=True, normalize_embeddings=True)
    eb = _umodel.encode(sent_b, convert_to_tensor=True, normalize_embeddings=True)
    return float(st_util.cos_sim(ea, eb))

def perplexity(text: str) -> float:
    try:
        return _ppl.compute(predictions=[text])["perplexities"][0]
    except Exception:
        return math.inf

def extract_item_list(response, target):
    try:
        response = response.replace(" ", " ")
        target = target.replace(" ", " ").replace("&amp;", "&").replace("&reg;","®")
        index = response.rfind(target)
        if index != -1:
            preceding_text = response[:index].strip()
            numbers = re.findall(r'\d+', preceding_text)
            if numbers:
                result_list = numbers
            else:
                result_list = []
        else:
            result_list = []
    except:
        result_list = []
    return result_list


def extract_ranking_and_explanations(response: str, raw_input: str = None):
    """
    Parse an LLM’s <ranking>…</ranking> block and return
    List[(item_idx:int, explanation:str), …].
    Supports:
      • 1. 5 – Explanation                (numeric index)
      • 1. "Title" – Explanation          (quoted title)
      • 1. ""Title"" – Explanation        (double-quoted)
      • 1. Title – Explanation            (unquoted title)
    """
    # 0) Normalize double‐double quotes → single quotes
    response = response.replace('""', '"')

    # 1) Extract the <ranking>…</ranking> block
    tag_re = re.compile(r'<\s*ranking\s*>(.*?)<\s*/\s*ranking\s*>',
                        re.DOTALL | re.IGNORECASE)
    m = tag_re.search(response)
    block = m.group(1).strip() if m else response

    # 2) Build index→{title,desc} map from the raw_input’s Candidate set
    meta = build_local_meta(raw_input) if raw_input else {}

    results = []
    # A) purely index-based, e.g. "1. 5 – Explanation"
    pat_idx    = re.compile(r'^\s*(\d+)[.)]\s*(\d+)\s*[-–—:]\s*(.*)$')
    # A2) index–title–explanation, e.g. "1. 3 – \"Babe\" – Explanation"
    pat_idx_title = re.compile(
        r'^\s*(\d+)[.)]\s*(\d+)\s*[-–—:]\s*"?(.+?)"?\s*[-–—:]\s*(.*)$'
    )
    # B) strict quoted title: '1. "Title" – Explanation'
    pat_qt     = re.compile(r'^\s*(\d+)[.)]\s*"([^"]+)"\s*[-–—:]\s*(.*)$')
    # C) loose title (no quotes): '1. Title – Explanation'
    pat_loose  = re.compile(r'^\s*(\d+)[.)]\s*(.+?)\s*[-–—:]\s*(.*)$')

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        # 0) New triple‑field pattern: index – title – explanation
        m_kt = pat_idx_title.match(line)
        if m_kt:
            _, item_idx, title, expl = m_kt.groups()
            results.append((int(item_idx), expl.strip()))
            continue

        # 1) index-based
        m_idx = pat_idx.match(line)
        if m_idx:
            _, item_idx, expl = m_idx.groups()
            results.append((int(item_idx), expl.strip()))
            continue

        # 2) strict quoted title
        m_qt = pat_qt.match(line)
        if m_qt:
            _, title, expl = m_qt.groups()
            title_clean = title.strip()
            # case‐insensitive lookup
            idx = next(
                (i for i,v in meta.items()
                   if v['title'].lower() == title_clean.lower()),
                None
            )
            if idx is not None:
                results.append((idx, expl.strip()))
            continue

        # 3) loose (unquoted) title
        m_ls = pat_loose.match(line)
        if m_ls:
            _, title, expl = m_ls.groups()
            title_clean = title.strip().strip('"')
            idx = next(
                (i for i,v in meta.items()
                   if v['title'].lower() == title_clean.lower()),
                None
            )
            if idx is not None:
                results.append((idx, expl.strip()))
            continue

        # 4) continuation of previous explanation
        if results:
            prev_idx, prev_expl = results[-1]
            results[-1] = (prev_idx, prev_expl + ' ' + line)

    return results


def detect_error(response: str, target: str, mode: str = 'improve') -> bool:
    """
    Returns False (i.e. “error”) when:
      1. We can’t parse any (rank, explanation) pairs from the response.
      2. OR the ground‑truth target isn’t in that parsed list.
      3. OR (in 'improve' mode) the target’s rank is >= threshold.

    Otherwise returns True (i.e. “no error”).
    """
    # 1) Try to parse the LM’s <ranking>…</ranking> output
    try:
        parsed = extract_ranking_and_explanations(response)
    except Exception:
        return False

    # If no items parsed, that’s a formatting error
    if not parsed:
        return False

    # 2) Find where our target lives in the parsed list
    #    parsed is List[ (item_idx:int, explanation:str), … ]
    ids = [item_idx for item_idx, _ in parsed]
    try:
        rank = ids.index(int(target)) + 1
    except ValueError:
        # target not in the parsed ranking
        return False

    # 3) In improve mode, treat anything ranked >= threshold as an error
    if mode == 'improve':
        threshold = 1
        return rank < threshold

    # 4) In select mode, we never treat it as “error”
    if mode == 'select':
        return True

    # Default fallback
    return False

def extract_edit_prompt(response):
    pattern = r'<START>\s*(.*?)\s*<END>'
    result_list = re.findall(pattern, response, re.DOTALL)
    if len(result_list) == 0:
        pattern = r'<START>(.*?)<END>'
        result_list = re.findall(pattern, response, re.DOTALL)
    return result_list 

def load_eval_data(config):
    with open(f"{config['data_path']}{config['dataset']}/Text/test_seed_{config['seed']}.json", 'r') as json_file:
        test_data = json.load(json_file)
    return test_data

# utils.py  (append at bottom)

import csv
from pathlib import Path
from datetime import datetime

class LogTable:
    """
    Minimal drop‑in replacement for wandb.Table.
    Keeps rows in memory and can dump to CSV at the end.
    """
    def __init__(self, columns, save_dir="logs", fname_prefix="table"):
        self.columns = columns
        self.rows = []
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fname = self.save_dir / f"{fname_prefix}_{ts}.csv"

    def add_data(self, *args):
        assert len(args) == len(self.columns), "column mismatch"
        self.rows.append(args)

    def save(self):
        with open(self.fname, "w", newline="", encoding="utf‑8") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            writer.writerows(self.rows)
        print(f"[LogTable] Saved {len(self.rows)} rows → {self.fname}")

# utils.py (append near bottom)  ──────────────────────────────────────
from sentence_transformers import SentenceTransformer, util

# load once – small, GPU‑optional model
_expl_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def explanation_similarity(expl: str, title: str, desc: str | None = None):
    """
    Cosine sim between explanation and item metadata.
    desc can be None or a long string – we truncate for efficiency.
    """
    meta = title if desc is None else (title + " " + desc[:160])  # ≤ ~256 tokens
    emb_e = _expl_model.encode(expl,  convert_to_tensor=True, normalize_embeddings=True)
    emb_m = _expl_model.encode(meta, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(emb_e, emb_m))

def build_local_meta(raw_input: str) -> dict[int, dict[str, str]]:
    """
    Extract metadata from the 'Candidate set:' block contained in a single JSON
    record.

    Returns
    -------
    {index:int -> {"title": str, "desc": str}}
        • `index`  is the 1‑based candidate ID that the prompt uses.
        • `title`  is the product title.
        • `desc`   is the raw tag string (can be an empty string).
    """
    try:
        block = raw_input.split("Candidate set:")[1]
    except IndexError:
        # no candidate block found – return empty dict
        return {}

    meta: dict[int, dict[str, str]] = {}

    # Regex explanation:
    #   (\d+)          : 1+ digits  (item index)
    #   \."([^"]+)"    : a dot, opening quote, then the title up to next quote
    #   \s*?\n\s*Tags: : newline + 'Tags:' label
    #   (.*?)\n        : non‑greedy match of everything until the next newline
    pattern = re.compile(
        r'\s+(\d+)\.\s*"([^"]+)"\s*\n\s*Tags:\s*([^\n]*?)\s*(?:\n|$)',
        flags=re.MULTILINE,
    )

    for m in pattern.finditer(block):
        idx   = int(m.group(1))
        title = m.group(2).strip()
        tags  = m.group(3).strip().rstrip(",")    # drop trailing comma
        meta[idx] = {"title": title, "desc": tags}

    return meta

def extract_between(text: str, marker: str = "##") -> str | None:
    """
    Returns the substring between the first pair of `marker`s.
    Example
    -------
    >>> extract_between("I say ##42##, matey!")
    '42'
    """
    import re, html
    pat = re.escape(marker) + r"(.*?)" + re.escape(marker)
    m = re.search(pat, text, flags=re.DOTALL)
    return html.unescape(m.group(1).strip()) if m else None


def parse_freeform_ranking(response: str):
    """Fallback parser for ranking output without special tags"""
    results = []
    # Define patterns separately for better readability
    patterns = [
        r'^\s*\d+[.)]\s*\d+\s*[-–—:.]\s*',  # 1. 123 - Explanation
        r'^\s*\(\d+\)\s*\d+\s*[-–—:.]\s*',   # (1) 123 - Explanation
        r'^\s*\d+\s*[-–]\s*\d+\s*[-–—:.]\s*', # 1- 123 - Explanation
        r'^\s*\d+\s*:\s*\d+\s*[-–—:.]\s*'    # 1: 123 - Explanation
    ]
    
    current_item = None
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
            
        matched = False
        for pat in patterns:
            if re.match(pat, line, re.IGNORECASE):
                try:
                    # Extract the item index which comes after the rank marker
                    parts = re.split(r'[-–—:.]', line, maxsplit=2)
                    if len(parts) < 2:
                        continue
                    
                    # Find the first number after the rank marker
                    num_match = re.search(r'\d+', parts[1])
                    if not num_match:
                        continue
                        
                    item_idx = int(num_match.group())
                    explanation = parts[2].strip() if len(parts) > 2 else ""
                    results.append((item_idx, explanation))
                    current_item = item_idx
                    matched = True
                    break
                except (ValueError, IndexError):
                    continue
        
        # Handle multi-line explanations
        if not matched and current_item is not None and results:
            # Continue previous explanation
            results[-1] = (current_item, results[-1][1] + " " + line)
    
    return results
def clean_llm_output(raw: str) -> str:
    # Handle Hugging Face pipeline output
    if raw.strip().startswith('[') and 'generated_text' in raw:
        try:
            data = ast.literal_eval(raw)
            if isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    return data[0].get('generated_text', raw)
                elif isinstance(data[0], str):
                    return data[0]
        except (SyntaxError, ValueError, TypeError):
            pass
    
    # Handle OpenAI-style JSON response
    if raw.strip().startswith('{') and ('content' in raw or 'text' in raw):
        try:
            data = json.loads(raw)
            return data.get('content', data.get('text', raw))
        except json.JSONDecodeError:
            pass
    
    # Remove common artifacts
    raw = re.sub(r'<\|[^|]+\|>', '', raw)  # OpenAI tokens
    raw = raw.replace("\\n", "\n")          # Unescape newlines
    raw = re.sub(r'\n{3,}', '\n\n', raw)    # Reduce excessive newlines
    
    return raw.strip()

def normalize_indices(pairs, candidate_size: int):
    """
    If it looks 0-based (0..candidate_size-1), shift to 1-based (1..candidate_size).
    """
    if not pairs:
        return pairs
    ids = [i for i,_ in pairs]
    if (0 in ids) or (max(ids) == candidate_size - 1 and candidate_size not in ids):
        return [(i + 1, expl) for i, expl in pairs]
    return pairs
