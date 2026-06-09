# A Collaborative Filtering Approach to Children's Book Recommendation

Chenyi Huang (Jenny) · Ke Xu (Kate) · CS 5100, Northeastern University

## Project Structure
```
├── eda_cleaning.ipynb    # EDA and data cleaning
├── model.ipynb           # Model training, tuning, and evaluation
├── recommend.py          # Console application
├── cleaned_ratings.csv   # Cleaned dataset (via Google Drive, see below)
├── requirements.txt
└── README.md
```
Raw data (not included) → place in parent directory:
`../goodreads_interactions_children.json.gz`
`../goodreads_books_children.json.gz`

## Data Setup

### Option A — Use pre-cleaned data (recommended for collaborators)
Download `cleaned_ratings.csv` directly from Google Drive:  
👉 **[https://drive.google.com/file/d/10Kpnn3AdDXqNDrKhH25Etjyw1GHHmIxI/view?usp=drive_link]**  
Place the file in the project root directory.

### Option B — Generate from raw data
1. Download the raw Goodreads Children dataset from:  
   https://mengtingwan.github.io/data/goodreads.html  
   - `goodreads_interactions_children.json.gz`  
   - `goodreads_books_children.json.gz`  
   Place both files in the **parent directory** of this project (i.e., one level up).

2. Run `eda_cleaning.ipynb` from start to finish (~10–15 min).  
   This will generate `cleaned_ratings.csv` in the project root.

3. Generate train/val/test split  
   Run the **"Train/Val/Test Split"** section in `model.ipynb`.  
   This produces `train_ratings.csv`, `val_ratings.csv`, `test_ratings.csv`  
   (requires `cleaned_ratings.csv` from step 2).

> **Note:** Raw data files and `cleaned_ratings.csv` are excluded from the GitHub repository (see `.gitignore`) due to file size constraints.

## Environment Setup
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
                              # Windows: venv\Scripts\activate

pip install -r requirements.txt

In VSCode: open any .ipynb → select kernel → Python (venv)
```

## How to Run

Run in the following order:

**1. EDA & Data Cleaning**
Open `eda_cleaning.ipynb` in VSCode, select the venv kernel, and run all cells.
*(Skip if you already have `cleaned_ratings.csv` from Google Drive.)*

**2. Model Training & Evaluation**
Open `model.ipynb` in VSCode, select the venv kernel, and run all cells.

**3. Console Application** 
python `recommend.py`

## Dataset
**Source:** UCSD Goodreads Children's Book Dataset  
https://mengtingwan.github.io/data/goodreads.html

| | Raw | Cleaned |
|---|---|---|
| Ratings | 6,384,470 | 4,044,839 |
| Books | 122,741 | 22,931 |
| Users | 462,164 | 61,078 |
| Sparsity | 99.99% | 99.71% |
| Fill rate | 0.0113% | 0.289% |

**Cleaning:** Three filters applied —
- Remove `rating = 0` (shelf-adds with no preference signal)
- Book density filter: keep books with ≥ 20 ratings
- User density filter: keep users with ≥ 20 ratings  
  *(Filters 2 & 3 applied iteratively until convergence — 14 iterations)*

**Output:** `cleaned_ratings.csv` · 170 MB · columns: `user_id, book_id, rating`

**Train/Val/Test Split (80/10/10, random_state=42):**  
Run the first section of `model.ipynb` to generate:  
`train_ratings.csv` · `val_ratings.csv` · `test_ratings.csv`

**Citation:** If using this dataset, please cite:  
Wan, M., & McAuley, J. (2018). RecSys '18. https://doi.org/10.1145/3240323.3240369  
Wan, M., et al. (2019). ACL '19. https://doi.org/10.18653/v1/P19-1248