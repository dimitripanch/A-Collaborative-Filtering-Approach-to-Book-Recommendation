"""Console book recommender.

Loads the trained SVD model and recommends books similar to one the user likes,
using cosine similarity between the books' learned latent-factor vectors (item
factors). This is item-to-item retrieval: it needs only a book, not a user profile.

Recommendations are collapsed by Goodreads `work_id` so that different editions of
the same work (which have distinct book_ids) appear only once.

Run:  python recommend.py
Requires:  svd_model.pkl, book_titles.csv, book_works.csv  (produced by model.ipynb)
"""

import pickle
import sys

import numpy as np
import pandas as pd

MODEL_PATH = "svd_model.pkl"
TITLES_PATH = "book_titles.csv"
WORKS_PATH = "book_works.csv"
TOP_N = 10


def load():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    titles = pd.read_csv(TITLES_PATH, dtype={"book_id": str})
    id_to_title = dict(zip(titles["book_id"], titles["title"]))

    # book_id -> work_id, to collapse duplicate editions of the same work
    works = pd.read_csv(WORKS_PATH, dtype={"book_id": str, "work_id": str})
    book_to_work = dict(zip(works["book_id"], works["work_id"]))

    # Map Surprise inner item ids <-> raw book_ids (as strings), and stack the
    # item-factor matrix in inner-id order so row i == model.qi[i].
    trainset = model.trainset
    inner_to_raw = {i: str(trainset.to_raw_iid(i)) for i in trainset.all_items()}
    raw_to_inner = {raw: i for i, raw in inner_to_raw.items()}

    qi = model.qi  # (n_items, n_factors)
    # Pre-normalize rows so cosine similarity is a single matrix-vector product.
    norms = np.linalg.norm(qi, axis=1, keepdims=True)
    qi_unit = qi / np.where(norms == 0, 1, norms)

    return model, id_to_title, inner_to_raw, raw_to_inner, qi_unit, book_to_work


def search_titles(query, id_to_title, raw_to_inner, book_to_work):
    """Case-insensitive substring match, restricted to books the model knows.
    Collapsed by work_id so each distinct work shows once (keeps the shortest title)."""
    q = query.strip().lower()
    hits = [
        (bid, title)
        for bid, title in id_to_title.items()
        if q in title.lower() and bid in raw_to_inner
    ]
    # Shorter titles first → exact-ish matches surface above long series entries.
    hits.sort(key=lambda x: len(x[1]))
    seen_works, deduped = set(), []
    for bid, title in hits:
        work = book_to_work.get(bid, bid)
        if work in seen_works:
            continue
        seen_works.add(work)
        deduped.append((bid, title))
    return deduped


def similar_books(book_id, id_to_title, inner_to_raw, raw_to_inner, qi_unit,
                  book_to_work, top_n=TOP_N):
    inner = raw_to_inner[book_id]
    sims = qi_unit @ qi_unit[inner]          # cosine similarity to every book
    order = np.argsort(-sims)                # most similar first
    # Collapse by work_id: skip the query's own work and any work already recommended.
    seen_works = {book_to_work.get(book_id, book_id)}
    out = []
    for j in order:
        if j == inner:
            continue                          # skip the query book itself
        raw = inner_to_raw[j]
        work = book_to_work.get(raw, raw)
        if work in seen_works:
            continue                          # already showed an edition of this work
        seen_works.add(work)
        out.append((id_to_title.get(raw, f"[{raw}]"), sims[j]))
        if len(out) == top_n:
            break
    return out


def main():
    print("Loading model...")
    model, id_to_title, inner_to_raw, raw_to_inner, qi_unit, book_to_work = load()
    print(f"Ready — {len(raw_to_inner):,} books available.\n")
    print("Type part of a book title to get similar recommendations (or 'q' to quit).\n")

    while True:
        query = input("Book you like > ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        if not query:
            continue

        hits = search_titles(query, id_to_title, raw_to_inner, book_to_work)
        if not hits:
            print("  No matching book found. Try a different title.\n")
            continue

        # Disambiguate when several titles match.
        if len(hits) > 1:
            print(f"  Found {len(hits)} matches:")
            for i, (_, title) in enumerate(hits[:10], 1):
                print(f"    {i}. {title}")
            choice = input("  Pick a number (Enter for 1) > ").strip()
            idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(hits[:10]) else 0
        else:
            idx = 0

        book_id, title = hits[idx]
        print(f"\n  Because you like \"{title}\":")
        for rank, (rec_title, sim) in enumerate(
            similar_books(book_id, id_to_title, inner_to_raw, raw_to_inner, qi_unit,
                          book_to_work), 1
        ):
            print(f"    {rank:>2}. {rec_title}   (similarity {sim:.3f})")
        print()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)
