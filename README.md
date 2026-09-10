# ExSBR: Enhancing Session-based Recommendations with Insightful Explanations via Self-Refinement Prompting

Official repository for the paper:

**ExSBR: Enhancing Session-based Recommendations with Insightful Explanations via Self-Refinement Prompting**  
  *IEEE International Conference on Data Mining (ICDM 2026)*

## Overview

ExSBR is an explainable session-based recommendation framework that combines topic-guided sample selection, self-refinement prompting, and LLM-based self-judgment to improve recommendation accuracy while generating concise and intent-aligned explanations.

## Data

Place the preprocessed datasets in ```./Dataset/```.

## Usage

### Prompt Refinement

Run ```python refine.py``` to perform prompt refinement and obtain the refined prompt used by ExSBR.

### Testing

Run ```python test.py``` to evaluate ExSBR on the test sessions.

### Evaluation

Testing outputs are saved in files such as ```eval_texts_xxx.csv```.

Run ```python eval_output.py``` on the saved evaluation outputs to compute the corresponding evaluation results.

## Citation

If you find this work useful, please consider citing our paper. The BibTeX entry will be updated after the ICDM 2026 proceedings become available.
