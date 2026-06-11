# SVD — Implementation Notes
### Project: Children's Book Recommender (CS 5100) — Jenny (TwoTower) + Kate (SVD)
*Library: Surprise (Hug 2020). Shares cleaned data, splits, and the cross-model eval pipeline with `twotower.ipynb`.*

> **Reading guide.**
> 1. Design decisions
> 2. Data preparation
> 3. Hyperparameter search → final model
> 4. Evaluation (RMSE + cross-model graded NDCG)
> 5. Core findings
> 6. Console application
> 7. Environment & reproducibility
> 8. Status & TODOs

---

# 1. Design Decisions

## 1.1 Why SVD (Surprise)
Classic Koren-style matrix factorization for **explicit ratings**. Chosen as the conventional
CF baseline against TwoTower. Bias terms (global + user + item) absorb the dataset's strong
positive skew (mean rating ≈ 3.99). Picked over SVD++ (too slow at 3.2M rows for marginal gain)
and KNN (no bias handling, worse on skewed data).

## 1.2 Target and metrics
- **Target = `rating`** (1–5): SVD is a rating-prediction model and requires it. This is also
  what forces `rating` (not implicit feedback) as the shared target for both models.
- **RMSE = SVD's own rating-task check.** Not the cross-model metric — TwoTower can't produce RMSE.
- **Cross-model metric = graded NDCG@10** on shared candidate pools (see §4.2), computed with the
  *same* functions as `twotower.ipynb` so the numbers are directly comparable.

## 1.3 Console recommendation = item-factor similarity
The console takes a book, not a user, so recommendation is **item-to-item**: cosine similarity
between the books' learned latent vectors (`model.qi`). No user profile needed (per-user fold-in
is a stretch goal). Trade-off: factors are trained for rating prediction, not retrieval, so
similarity quality is good for high-signal books and noisier for sparse ones (§5.2).

---

# 2. Data Preparation

Uses `cleaned_ratings.csv` and the 80/10/10 split unchanged (shared with Jenny → controls the
"data" variable so SVD-vs-TwoTower is fair). 3,235,873 train / 404,483 val / 404,483 test.
**Zero cold-start** (every val/test user and book appears in train), so SVD never hits an unknown
id. Val is unused on the SVD line — Surprise's `GridSearchCV` does its own internal K-fold.

---

# 3. Hyperparameter Search → Final Model

*Method:* `GridSearchCV`, 3-fold CV, RMSE objective, `n_jobs=-1`. A single SVD fit on full train
is only ~8–28s, so full-data grid search is feasible (~36 min for the final grid).

## 3.1 Search iterations (the methodology matters here)
| # | data used | grid | best params | note |
|---|---|---|---|---|
| 1 | 1M subsample | wide | nf=50, reg=0.1 | **rejected** — subsample biases toward small/over-regularized models |
| 2 | full train | grid shifted *up* (dropped baseline's lr=0.005, reg=0.02) | nf=150 | **rejected** — grid excluded the baseline region; tuned RMSE 0.7631 *worse* than default |
| 3 | full train | **centered** on defaults, extended outward | **nf=200, ep=40, lr=0.02, reg=0.1** | **final** |

**Key lesson:** optimal hyperparameters shift with data size — the 1M subsample picked nf=50,
full train picked nf=200 (more data supports more capacity). Tuning on a subsample would have
locked in the wrong complexity, so the final search uses the full train set.

## 3.2 FINAL grid (locked)
`n_factors ∈ {100,150,200}, n_epochs ∈ {20,30,40}, lr_all ∈ {0.005,0.01,0.02}, reg_all ∈ {0.02,0.05,0.1}`
(81 combos × cv=3 = 243 fits). Centered on the Surprise defaults and extended outward so the
optimum is bracketed, not on a boundary. Best CV RMSE ≈ 0.7573–0.7576 (4th-decimal jitter from
`n_jobs=-1` float-accumulation order; selected params identical across runs).

## 3.3 Final model
`SVD(n_factors=200, n_epochs=40, lr_all=0.02, reg_all=0.1, random_state=42)`, refit on full train,
saved to `svd_model.pkl`.

---

# 4. Evaluation

## 4.1 RMSE (rating task)
| Model | test RMSE |
|---|---|
| Baseline (defaults: nf=100, ep=20, lr=0.005, reg=0.02) | **0.7614** |
| Tuned (nf=200, ...) | **0.7627** |

Reference: predicting the global mean gives RMSE ≈ 0.937 (the rating std), so SVD explains real
signal. Tuned ≈ baseline (Δ0.0013, noise).

## 4.2 Cross-model graded NDCG@10
Reuses Jenny's `build_candidate_pools` (seed=42, each user = held-out test books as graded
positives + 100 sampled negatives) and `evaluate_ndcg` (`sklearn.metrics.ndcg_score`), so pools
are byte-identical to the TwoTower run. SVD score_fn = predicted rating (`.est`).

| Model | graded NDCG@10 | test RMSE |
|---|---|---|
| SVD default | 0.1587 | 0.7614 |
| **SVD tuned (ours)** | **0.1628** (58,787 users) | 0.7627 |
| TwoTower (Jenny) | 0.8508 | — |

---

# 5. Core Findings

## 5.1 Tuning gives no meaningful gain — the model is at its error floor
Both metrics agree: tuned ≈ default (RMSE 0.7627 vs 0.7614; NDCG 0.1628 vs 0.1587). On full train
the CV objective keeps wanting more capacity (n_factors hits the grid ceiling), but the extra
capacity does *not* transfer to the held-out test set. Cause: the bias terms already explain most
predictable variance; the latent factors only model a small residual, and ratings carry
irreducible subjective noise. **Honest takeaway: default SVD is already near-optimal here;
complexity ≠ better.** (Mirrors Jenny's TwoTower finding that embed=32 and default lr won.)

## 5.2 RMSE–NDCG divergence (the key conceptual finding)
Same model: **good at rating prediction (RMSE 0.76), poor at ranking (NDCG 0.16).** SVD's
predictions cluster near the global mean (~3.8) because 72% of ratings are 4–5★, so positives and
negatives are hard to separate when the predicted rating is used as a ranking score — and NDCG is
sensitive to ordering. **Low RMSE does not imply good top-K ranking; the two measure different
capabilities.** Each model is strong on its designed task (SVD: rating; TwoTower: ranking); on the
recommendation-as-ranking task the neural model fits far better. *(See Jenny's §5.3–5.4: the gap
is architectural, not just a usage artifact.)*

---

# 6. Console Application (`recommend.py`)

Input a book title → top-N similar books by item-factor cosine similarity. Supports fuzzy
(substring) title search with disambiguation when several titles match.

**work_id de-duplication (display layer).** Goodreads gives each *edition* its own `book_id`
(22,931 books → 20,471 works; ~16.1% of the catalog is multi-edition), so raw output repeats
titles. We collapse search results and recommendations by `work_id` (a `book_id→work_id` map built
in `model.ipynb §6`, O(1) lookup, no speed impact). **Decision: dedup at display, not in cleaning**
— a data-layer merge would force re-clean/re-split/retrain both models, and the impact on
evaluation/training is measured to be negligible (per Jenny's §7). Stated as a limitation.

Example: *Magic Tree House: #1-4* → #5-8 (0.88), #9-12 (0.70), #17-24 (0.64) — series captured well.

---

# 7. Environment & Reproducibility

- Python 3.14, project `.venv` (Homebrew Python is PEP-668 externally-managed → venv required).
- `scikit-surprise 1.1.5` (cp314 wheel, no compile), `scikit-learn`, `pandas`, `numpy`.
- Pipeline: `eda_cleaning.ipynb` → `cleaned_ratings.csv` → `model.ipynb` (§1 split → §3 baseline →
  §4 GridSearchCV → §5 final model → §6 title/work maps → §7 graded NDCG) → `recommend.py`.
- `model.ipynb` runs top-to-bottom in one session (execution counts 1–8, all outputs present).
  Fixed seeds throughout (random_state=42).

---

# 8. Status & TODOs

## Done
- Baseline SVD (RMSE 0.7614)
- GridSearchCV on full train, final nf=200 (RMSE 0.7627)
- Cross-model graded NDCG@10 = 0.1628 (shared pools with TwoTower)
- Console app + work_id display de-dup
- Requirements.txt updated (scikit-learn)

## TODOs
- Write report sections: tuning methodology (subsample→full, boundary bracketing), RMSE–NDCG
  divergence, console + work_id limitation.
- Decide stretch goals (fold-in for per-user recs has more depth than a GUI).
- Optional: algorithm comparison (NMF/KNN) if time allows.
