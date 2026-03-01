# CollabRec – A Simple Recommendation System

CollabRec is a program that recommends items to people based on what similar people or items they liked. It uses a technique called collaborative filtering.

## Requirements

- Python 3.6 or higher
- No extra packages – it uses only the standard library.

## Installation

1. Download or clone the `collabrec` folder.
2. Place it anywhere on your computer.
3. Make sure the folder structure is like:

```
collabrec/
├── collabrec/          (the main package)
│   ├── __init__.py
│   ├── constants.py
│   ├── data/
│   ├── similarity/
│   ├── models/
│   ├── evaluation/
│   └── cli/
├── ratings.csv         (your data file – you can place it here)
└── generate_ratings.py (This is already created, you can update the code if you want)
```

## What It Does

- **Give recommendations** – for any user, it lists items they haven't seen, ranked by how much they would probably like them.
- **Predict a single rating** – you can ask "what would user X give to item Y?"
- **Show similar items** – given an item, it lists other items that are rated similarly by users.
- **Show similar users** – given a user, it lists other users with similar taste.
- **Evaluate accuracy** – it can test itself by hiding some ratings and seeing how well it predicts them. It reports RMSE and MAE (error numbers).
- **Show dataset statistics** – tells you how many users, items, ratings, etc.

## How It Works

1. **Read the data** – from a CSV file with columns `user_id, item_id, rating`.
2. **Build a model** – either item‑based or user‑based:
   - **Item‑based**: it looks at how similar items are. If you liked "Inception", you might also like "Interstellar" because they are similar.
   - **User‑based**: it looks at how similar users are. If Alice and Bob like the same movies, then Bob's likes can be recommended to Alice.
3. **Make predictions** – using a weighted average of neighbours' ratings.
4. **Give recommendations** – predict for all unseen items and pick the highest scoring ones.

The similarity between two users or items can be measured with **cosine similarity** or **Pearson correlation**. You can also use adjusted cosine for item‑based.

## How to Use

First, open a terminal and go to the folder that contains the `collabrec` directory (not inside it). For example:

```bash
cd /home/yourname/Projects
```

### Generate a sample dataset (if you don't have one)

```bash
cd collabrec
python3 generate_ratings.py
cd ..
```

This creates `collabrec/ratings.csv` with about 50,000 ratings.

### Get recommendations for a user

```bash
python3 -m collabrec.cli.main --data collabrec/ratings.csv --user user_0001 --top 5
```

This gives the top 5 items for user `user_0001`.

### Predict a single rating

```bash
python3 -m collabrec.cli.main --data collabrec/ratings.csv --predict user_0001 item_0050
```

### Find similar items

```bash
python3 -m collabrec.cli.main --data collabrec/ratings.csv --similar-items item_0010 --top 5
```

### Evaluate accuracy (random split)

```bash
python3 -m collabrec.cli.main --data collabrec/ratings.csv --eval --eval-split 0.8
```

This uses 80% of ratings to train and 20% to test.

### Show dataset statistics

```bash
python3 -m collabrec.cli.main --data collabrec/ratings.csv --stats
```

### Save output as JSON

Add `--json filename.json` to any command that produces output (recommendations, evaluation, etc.)

## Options You Can Change

- `--method` : `item` or `user` (default item)
- `--similarity` : `cosine` or `pearson` (default cosine)
- `--k` : number of neighbours (default 20)
- `--min-common` : minimum common ratings to count similarity (default 2)
- `--adjusted-cosine` : use adjusted cosine (item‑based only)
- `--seed` : random seed for reproducible splits (default 42)

For a full list:

```bash
python3 -m collabrec.cli.main --help
```

## Example Output

```
  Loaded 24757 ratings | 500 users | 100 items
  Model trained in 0.524s | ItemBasedCF(similarity='cosine', k=20, min_common=2, fitted)

========================================================
       Top‑5 Recommendations for 'user_0001'
========================================================
  Method     : item
  Similarity : cosine

  Rank   Item                           Pred. Score   Confidence
  ────────────────────────────────────────────────────────────
  1      item_0234                      4.7           ████████░░ 80%
  2      item_0089                      4.5           ███████░░░ 70%
  3      item_0456                      4.3           ██████░░░░ 60%
  4      item_0123                      4.1           █████░░░░░ 50%
  5      item_0789                      3.9           ████░░░░░░ 40%
```

## What the Numbers Mean

- **Predicted Score** – the estimated rating (on the same scale as your data, e.g., 1‑5).
- **Confidence** – how many of the neighbours were actually used. Higher means more reliable.
- **RMSE / MAE** – error measures; lower is better. RMSE gives more weight to big errors.

## Possible Use Cases

- **Movie or book recommendations** – like "people who liked this also liked…"
- **E‑commerce** – suggest products based on purchase history.
- **Music playlists** – recommend songs similar to the ones you already enjoy.
- **Experimental** – try it on any dataset where users rate things (e.g., restaurant reviews, course ratings). It's small enough to experiment with.

## License

This project is for educational purposes and is free to use under the MIT license.

## Need Help?

If you have questions, feel free to open an issue on the project page (if hosted on GitHub) or contact the author. Happy recommending!
