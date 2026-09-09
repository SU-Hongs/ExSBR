import os
# import wandb
from opt.eval import Eval
from opt.config import init_config
from opt.utils import load_eval_data
from opt.utils import LogTable
from opt.utils  import make_run_log 

if __name__ == '__main__':
    test_prompt = """You are a session-based recommendation engine with built-in explainability.
INPUT  
• A JSON object containing:  
  - “Current session interactions”: a list of items the user has just interacted with.  
  - “Candidate set”: exactly 20 items (each with title and tags). The input order (1-20) defines your **baseline initial ranking**.
TASK  
1. Detect the key themes or genres in the current session interactions.  
2. Infer the user's preferences from those themes.  
3. Treat the candidate set's input order as your baseline initial ranking **and** generate a one-sentence explanation for each item linking it to the inferred preferences.  
4. Use an LLM-based explanation model to produce a **new ranking** of the same 20 items, again giving each item a one-sentence justification.  
5. **Act as an independent recommendation judge:** without regard to which list came first, compare the two ranked lists **solely** on which best predicts the user's likely next interaction—i.e. which list places the most contextually appropriate items earlier based on the session context and inferred preferences. Discard the other and return only the superior ranking.
**OUTPUT:** exactly 20 lines, each formatted as:
<rank>. <original-index> - <item-title> - <explanation>
wrapped between `<ranking>` and `</ranking>`.
---
### Example1 (Re-Ranking is better)
Current session interactions:
[
"CAcafe Coconut Coffee, Coconut Infused Colombian Coffee, Creamy Drink Mix, 19.05oz"
]
**Original Ranking**
<ranking>
1. 1 – Healthworks Cacao Powder… – A healthy superfood choice, but no coconut flavor.
2. 2 – Healthworks Raw Goji Berries… – Tasty superfood, but still no coconut.
3. 3 – CAcafe Coconut Coffee… – Contains the exact coconut taste you just viewed.
…
20. 20 – Nutiva Organic Hemp Seed… – Least related to your coconut interest.
</ranking>
**Re-Ranking**
<ranking>
1. 3 – CAcafe Coconut Coffee, Coconut Infused Colombian Coffee… – Matches your coconut-coffee pick perfectly.
2. 1 – Healthworks Cacao Powder (32 Ounces)… – Next best with health benefits.
3. 2 – Healthworks Raw Goji Berries (8 Ounces)… – Still a good superfood snack.
…
20. 20 – Nutiva Organic Hemp Seed… – Least connected to coconut.
</ranking>
**Re-Ranking wins** because its order places more contextually relevant items (including the session item) in the top positions.
**Final Output:**  
<ranking>  
1. 3 - CAcafe Coconut Coffee, Coconut Infused Colombian Coffee, 19.05oz - Directly matches the session's coconut theme.  
2. 1 - Healthworks Cacao Powder (32 Ounces)… - Next best superfood match.  
…  
20. 20 - Nutiva Organic Hemp Seed…  
</ranking>
---
### Example2 (Original is better)
Current session interactions:
[
"HARIBO Gummi Candy, Original Goldbears, 8 oz. Bag"
]
**Original Ranking**
<ranking>
1. 1 – HARIBO Gummi Candy… – Exactly what you just picked.
2. 2 – OREO Double Stuf Chocolate Sandwich Cookies… – Another sweet cookie option.
3. 3 – Snack Pack Chocolate Pudding Cups… – A simple dessert snack.
…
20. 20 – Green Giant French Style Green Beans… – Least on-brand for sweets.
</ranking>
**Re-Ranking**
<ranking>
1. 2 – OREO Double Stuf Chocolate Sandwich Cookies… – Moves Oreo ahead of your exact pick.
2. 1 – HARIBO Gummi Candy… – Pushes your candy choice down.
3. 3 – Snack Pack Chocolate Pudding Cups… – Third place.
…
20. 20 – Green Giant French Style Green Beans…
</ranking>
**Original wins** because it preserves the strongest matches at the top without demoting key favorites.
**Final Output:**  
<ranking>  
1. 1 - HARIBO Gummi Candy, Original Goldbears, 8 oz. Bag - Exact match to session item.  
2. 2 - OREO Double Stuf Chocolate Sandwich Cookies, 12-15.35 oz - Next best sweet.  
…  
20. 20 - Green Giant French Style Green Beans, 14.5 Ounce Can - Least aligned.  
</ranking>
"""


    
    conf = init_config()
    conf["bad_log_fp"] = make_run_log()
    test_data = load_eval_data(conf)
    for data in test_data:
        data['target_index'] += 1

    

        # === logging ===
    if conf['backend'] == 'openai':
        openai_key = conf['openai_api_key']          # keep a copy
    else:
        openai_key = None                            # key absent for local

    # always log locally
    text_table = LogTable(
        columns=["Input", "Target", "Response", "Valid"],
        save_dir="logs",
        fname_prefix="eval_texts",
    )

    if openai_key is not None:                       # restore for internal use
        conf['openai_api_key'] = openai_key


    eval_model = Eval(conf, test_data, text_table)
    _results, target_rank_list, error_list = eval_model.run(test_prompt)
    text_table.save()
    

