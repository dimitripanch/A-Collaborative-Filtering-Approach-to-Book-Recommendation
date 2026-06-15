# recommend.py — Usage

## Purpose
A console book recommender: given one (or a few) books you like, it suggests
similar books. Built to **compare the SVD and TwoTower models** qualitatively,
alongside the quantitative NDCG results.

## Logic
Item-to-item retrieval: every book has a learned vector, and recommendations are
the books with the highest **cosine similarity** to the input.

- `--model svd` — uses the SVD item factors `qi` (trained for rating prediction;
  retrieval is a borrowed use).
- `--model twotower` — uses the TwoTower item-tower embeddings (trained directly
  for ranking/retrieval; better suited to this task).
- **Multi-seed** (twotower only): 2–5 book vectors are averaged into a
  "pseudo-user" centroid, then retrieved against.
- Results are collapsed by `work_id` (different editions of the same work appear
  once) and the seed books themselves are excluded.

## Usage
```bash
python recommend.py --model twotower    # recommended
python recommend.py --model svd
```
At the `Book(s) you like >` prompt, type:

- **One book** → item-to-item. e.g. `where the wild things are`
- **2–5 books** → comma-separated (ASCII `,` or full-width `，` both work),
  blends their tastes. e.g. `matilda, the bfg, holes`
- Titles are case-insensitive and match on substrings. When several books match,
  a single seed lets you pick a number; multi-seed auto-picks the shortest title.
- Type `q` to quit.

> Note: similarity scores are comparable **within** a model, not across them —
> don't compare SVD's 0.3 against TwoTower's 0.9 directly.

## Requires
| File | Needed for | Source |
|---|---|---|
| `recommend.py` | — | repo |
| `book_titles.csv` | both models | repo |
| `book_works.csv` | both models | repo |
| `twotower_item_embeds.npy` | `--model twotower` | repo |
| `twotower_book_ids.csv` | `--model twotower` | repo |
| `svd_model.pkl` | `--model svd` | generated locally — see below (too large for GitHub) |

`--model twotower` runs straight from a clone. `--model svd` additionally needs
`svd_model.pkl`, which is not committed (~217 MB, over GitHub's limit).

**Generating `svd_model.pkl`:** run **Section 5 "Final Model & Evaluation"** in
`model.ipynb` (it loads `best_params.json`, fits once, and saves the file). This
is a single fit — only **tens of seconds**. Skip Section 4 (GridSearchCV, ~36 min);
that is hyperparameter tuning and is not needed, since the best params are already
saved in `best_params.json`.

## Suggested demo books (confirmed in the catalog)
| Category | Examples |
|---|---|
| Classic chapter / fantasy | The Secret Garden · Peter Pan · Alice's Adventures in Wonderland · The Wind in the Willows · The Phantom Tollbooth |
| Roald Dahl | Matilda · The BFG · James and the Giant Peach · Charlie and the Chocolate Factory |
| Dr. Seuss / picture books | Green Eggs and Ham · The Cat in the Hat · The Lorax · Where the Wild Things Are · Goodnight Moon |
| Series | Magic Tree House · Percy Jackson · Diary of a Wimpy Kid |
| Poetry (Shel Silverstein) | A Light in the Attic · Where the Sidewalk Ends |

⚠️ This is the Goodreads **Children's** dataset, so adult literature is absent:
the English Harry Potter originals, *The Old Man and the Sea*, and Jane Austen's
novels won't be found (Austen exists only as children's adaptations like Cozy Classics).

## Example output

**TwoTower · single seed** — `where the wild things are`
```
Because you like "Where the Wild Things Are":
   1. A Light in the Attic              (similarity 0.934)
   2. The Phantom Tollbooth             (similarity 0.904)
   3. Falling Up                        (similarity 0.903)
   4. Goodnight Moon                    (similarity 0.894)
   5. The Lorax                         (similarity 0.892)
   6. The Sneetches and Other Stories   (similarity 0.886)
   7. James and the Giant Peach         (similarity 0.876)
   8. Are You There God? It's Me, Margaret  (similarity 0.871)
   9. Where the Sidewalk Ends           (similarity 0.865)
  10. The Indian in the Cupboard        (similarity 0.860)
```

**TwoTower · multi-seed (centroid)** — `matilda, the bfg, james and the giant peach`
```
Because you like these 3 books:
  • Matilda
  • The BFG
  • James and the Giant Peach

   1. Fantastic Mr. Fox                 (similarity 0.938)
   2. Danny the Champion of the World   (similarity 0.930)
   3. Bridge to Terabithia              (similarity 0.904)
   4. The Indian In The Cupboard        (similarity 0.895)
   5. Here's to You, Rachel Robinson    (similarity 0.894)
   6. Charlie and the Great Glass Elevator  (similarity 0.894)
   7. Charlie and the Chocolate Factory (similarity 0.893)
   8. The Magic Finger                  (similarity 0.889)
   9. Le Petit Prince                   (similarity 0.881)
  10. Oh, The Places You'll Go!         (similarity 0.860)
```
*(Three Roald Dahl seeds → mostly Dahl back — the centroid captures the taste well.)*

**SVD · single seed (for contrast)** — `where the wild things are`
```
Because you like "Where the Wild Things Are":
   1. The Ballad of Valentine           (similarity 0.360)
   2. The Snowy Day                     (similarity 0.350)
   3. King Bidgood's in the Bathtub     (similarity 0.349)
   4. Tikki Tikki Tembo                 (similarity 0.320)
   5. Frog and Toad Are Friends         (similarity 0.316)
   ...
```
*(Same book, but SVD's similarities are lower and the theme looser — consistent
with SVD trailing TwoTower on NDCG.)*
