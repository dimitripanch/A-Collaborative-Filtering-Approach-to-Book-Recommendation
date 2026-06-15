"""Console book recommender — SVD or TwoTower, selected with --model.

    python recommend.py --model svd
    python recommend.py --model twotower

Both modes do item-to-item retrieval: cosine similarity between books' learned
vectors (SVD item factors `qi`, or the TwoTower item-tower embeddings). They need
only a book, not a user profile.

  • SVD mode       — type ONE book title; recommends similar books.
  • TwoTower mode  — type ONE title for item-to-item, OR 2-5 comma-separated titles
                     to seed a "pseudo-user": the seed vectors are averaged into a
                     centroid (query-by-example) and the nearest books returned.

Recommendations are collapsed by Goodreads `work_id` so different editions of the
same work appear only once; all seed works are also excluded from the results.

Note: similarity scores are comparable WITHIN a model, not across SVD vs TwoTower.

Requires:
  always       — book_titles.csv, book_works.csv
  --model svd        — svd_model.pkl
  --model twotower   — twotower_item_embeds.npy, twotower_book_ids.csv
(all produced by model.ipynb / twotower.ipynb)
"""

import argparse
import pickle
import sys

import numpy as np
import pandas as pd

TITLES_PATH = "book_titles.csv"
WORKS_PATH = "book_works.csv"
SVD_MODEL_PATH = "svd_model.pkl"
TT_EMBED_PATH = "twotower_item_embeds.npy"
TT_IDS_PATH = "twotower_book_ids.csv"

TOP_N = 10
MAX_SEEDS = 5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_meta():
    """Title and work-id lookups, shared by both models."""
    titles = pd.read_csv(TITLES_PATH, dtype={"book_id": str})
    id_to_title = dict(zip(titles["book_id"], titles["title"]))
    works = pd.read_csv(WORKS_PATH, dtype={"book_id": str, "work_id": str})
    book_to_work = dict(zip(works["book_id"], works["work_id"]))
    return id_to_title, book_to_work


def _normalize_rows(mat):
    """L2-normalise each row so cosine similarity is a single dot product.
    Done defensively even if the source vectors are already unit length."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1, norms)


def load_svd():
    """SVD item factors. raw_to_idx maps book_id -> row in the unit matrix."""
    with open(SVD_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    trainset = model.trainset
    idx_to_raw = {i: str(trainset.to_raw_iid(i)) for i in trainset.all_items()}
    raw_to_idx = {raw: i for i, raw in idx_to_raw.items()}
    vecs_unit = _normalize_rows(model.qi)
    return raw_to_idx, idx_to_raw, vecs_unit


def load_twotower():
    """TwoTower item-tower embeddings; row i == book_ids[i] (aligned export)."""
    mat = np.load(TT_EMBED_PATH).astype(np.float64)
    ids = pd.read_csv(TT_IDS_PATH, dtype={"book_id": str})["book_id"].tolist()
    if len(ids) != mat.shape[0]:
        sys.exit(f"Misaligned export: {mat.shape[0]} vectors vs {len(ids)} book_ids.")
    idx_to_raw = dict(enumerate(ids))
    raw_to_idx = {raw: i for i, raw in idx_to_raw.items()}
    vecs_unit = _normalize_rows(mat)
    return raw_to_idx, idx_to_raw, vecs_unit


# --------------------------------------------------------------------------- #
# Search & retrieval (model-agnostic — operates on raw_to_idx / idx_to_raw / vecs)
# --------------------------------------------------------------------------- #
def search_titles(query, id_to_title, raw_to_idx, book_to_work):
    """Case-insensitive substring match, restricted to books the model knows,
    collapsed by work_id (keeps the shortest title so exact-ish matches surface)."""
    q = query.strip().lower()
    hits = [
        (bid, title)
        for bid, title in id_to_title.items()
        if q in title.lower() and bid in raw_to_idx
    ]
    hits.sort(key=lambda x: len(x[1]))
    seen_works, deduped = set(), []
    for bid, title in hits:
        work = book_to_work.get(bid, bid)
        if work in seen_works:
            continue
        seen_works.add(work)
        deduped.append((bid, title))
    return deduped


def rank_by_vector(qvec, vecs_unit, idx_to_raw, id_to_title, book_to_work,
                   exclude_idx, exclude_works, top_n=TOP_N):
    """Top-N books by cosine to qvec, skipping seed rows and any work already seen."""
    sims = vecs_unit @ qvec
    order = np.argsort(-sims)
    seen_works = set(exclude_works)
    out = []
    for j in order:
        if j in exclude_idx:
            continue
        raw = idx_to_raw[j]
        work = book_to_work.get(raw, raw)
        if work in seen_works:
            continue
        seen_works.add(work)
        out.append((id_to_title.get(raw, f"[{raw}]"), sims[j]))
        if len(out) == top_n:
            break
    return out


def recommend(seed_ids, vecs_unit, raw_to_idx, idx_to_raw, id_to_title, book_to_work):
    """One seed -> item-to-item; many seeds -> centroid (averaged unit vectors)."""
    idxs = [raw_to_idx[b] for b in seed_ids]
    if len(idxs) == 1:
        qvec = vecs_unit[idxs[0]]
    else:
        centroid = vecs_unit[idxs].mean(axis=0)
        qvec = centroid / max(np.linalg.norm(centroid), 1e-12)
    exclude_works = {book_to_work.get(b, b) for b in seed_ids}
    return rank_by_vector(qvec, vecs_unit, idx_to_raw, id_to_title, book_to_work,
                          exclude_idx=set(idxs), exclude_works=exclude_works)


# --------------------------------------------------------------------------- #
# Seed resolution (title string -> book_id)
# --------------------------------------------------------------------------- #
def resolve_interactive(query, id_to_title, raw_to_idx, book_to_work):
    """Single-seed: disambiguate interactively when several works match."""
    hits = search_titles(query, id_to_title, raw_to_idx, book_to_work)
    if not hits:
        return None
    if len(hits) > 1:
        print(f"  Found {len(hits)} matches:")
        for i, (_, title) in enumerate(hits[:10], 1):
            print(f"    {i}. {title}")
        choice = input("  Pick a number (Enter for 1) > ").strip()
        idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(hits[:10]) else 0
    else:
        idx = 0
    return hits[idx]


def resolve_auto(query, id_to_title, raw_to_idx, book_to_work):
    """Multi-seed: auto-pick the best (shortest-title) match, no prompt."""
    hits = search_titles(query, id_to_title, raw_to_idx, book_to_work)
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Console book recommender (SVD or TwoTower).")
    ap.add_argument("--model", required=True, choices=["svd", "twotower"],
                    help="which trained model to recommend from")
    args = ap.parse_args()

    print(f"Loading {args.model} model...")
    try:
        id_to_title, book_to_work = load_meta()
        if args.model == "svd":
            raw_to_idx, idx_to_raw, vecs_unit = load_svd()
        else:
            raw_to_idx, idx_to_raw, vecs_unit = load_twotower()
    except FileNotFoundError as e:
        sys.exit(f"Missing required file: {e.filename}")

    print(f"Ready — {len(raw_to_idx):,} books available ({args.model}).\n")
    if args.model == "twotower":
        print("Type a book title for similar books, OR 2-5 titles separated by commas")
        print("to blend them into a 'pseudo-user' recommendation. ('q' to quit.)\n")
    else:
        print("Type part of a book title to get similar recommendations (or 'q' to quit).\n")

    while True:
        raw = input("Book(s) you like > ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        if not raw:
            continue

        queries = [s.strip() for s in raw.split(",") if s.strip()]

        # --- enforce per-model seed-count rules ---
        if args.model == "svd" and len(queries) > 1:
            print("  Multi-seed is only supported in --model twotower. Enter one title.\n")
            continue
        if len(queries) > MAX_SEEDS:
            print(f"  Too many books ({len(queries)}). Use at most {MAX_SEEDS}.\n")
            continue

        # --- resolve titles -> book_ids ---
        if len(queries) == 1:
            hit = resolve_interactive(queries[0], id_to_title, raw_to_idx, book_to_work)
            seeds = [hit] if hit else []
            if not hit:
                print("  No matching book found. Try a different title.\n")
                continue
        else:
            seeds = []
            for q in queries:
                hit = resolve_auto(q, id_to_title, raw_to_idx, book_to_work)
                if hit:
                    seeds.append(hit)
                else:
                    print(f"  Skipped \"{q}\" — no match found.")
            if not seeds:
                print("  None of those titles matched. Try again.\n")
                continue
            if len(seeds) == 1:
                print("  Only one title matched — recommending from that single book.")

        seed_ids = [bid for bid, _ in seeds]
        seed_titles = [title for _, title in seeds]

        # --- header ---
        if len(seed_ids) == 1:
            print(f"\n  Because you like \"{seed_titles[0]}\":")
        else:
            print(f"\n  Because you like these {len(seed_ids)} books:")
            for t in seed_titles:
                print(f"    • {t}")
            print()

        # --- recommend ---
        recs = recommend(seed_ids, vecs_unit, raw_to_idx, idx_to_raw,
                         id_to_title, book_to_work)
        for rank, (rec_title, sim) in enumerate(recs, 1):
            print(f"    {rank:>2}. {rec_title}   (similarity {sim:.3f})")
        print()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)
