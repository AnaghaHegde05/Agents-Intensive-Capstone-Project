# BookNerd 🕯️

A multi-agent book recommendation system that uses text embeddings and
cosine similarity to match books to your taste — and learns from the
books you save as favorites.

## How it works

Four agents work together:

- **ProfileAnalyzerAgent** – turns your selected genres (and any saved
  favorites) into a single "taste vector"
- **SearchDataAgent** – queries the Google Books API for candidate books,
  with retry/backoff on transient failures
- **CriticAgent** – encodes each book's title/description/subjects with a
  sentence-transformer model and ranks books by cosine similarity to your
  taste vector
- **Orchestrator** – runs the full pipeline: analyze → search → filter by
  year → score → return top results

Saving a book to Favorites generates and stores its embedding, and future
recommendations blend it into your taste vector — so the system adapts
over time without any retraining.

An embedding cache avoids re-encoding books that have already been seen
in a previous search.

## Setup

```bash
pip install gradio requests sentence-transformers scikit-learn
```

Set a Google Books API key (optional but recommended to avoid rate
limits on the public quota):

```bash
export GOOGLE_BOOKS_API_KEY="your-key-here"
```

Run:

```bash
python booknerd.py
```

## Tech stack

Python, Gradio, sentence-transformers (`all-MiniLM-L6-v2`), scikit-learn,
Google Books API

## Known limitations

- Recommendations are stored in a single local JSON file (no multi-user
  session isolation)
- Book selection in the UI is matched by title+year label rather than a
  stable ID
