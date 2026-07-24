# -*- coding: utf-8 -*-
"""
BookNerd - Multi-Agent Book Recommendation System

A rule-based multi-agent pipeline that recommends books using semantic
embeddings and cosine similarity, with a feedback loop that updates the
user's taste vector based on saved favorites.

Agents:
  - ProfileAnalyzerAgent: builds a taste vector from selected genres + favorites
  - SearchDataAgent: queries the Google Books API
  - CriticAgent: scores/ranks books using embedding similarity
  - Orchestrator: coordinates the agents end-to-end
"""

# ==============================================
# 1. INSTALL DEPENDENCIES
# ==============================================
# pip install --quiet gradio requests sentence-transformers scikit-learn

# ==============================================
# 2. SET YOUR API KEY HERE
# ==============================================
import os

# --- Option A: Hardcode (quick, not recommended for shared/public code) ---
# os.environ['GOOGLE_BOOKS_API_KEY'] = 'YOUR_API_KEY_HERE'

# --- Option B: Colab Secrets (secure) ---
# from google.colab import userdata
# os.environ['GOOGLE_BOOKS_API_KEY'] = userdata.get('GOOGLE_BOOKS_API_KEY')

# ==============================================
# 3. IMPORTS
# ==============================================
import json
import requests
import re
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================
# 4. LOAD AI MODEL (once)
# ==============================================
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384  # Dimension of the all-MiniLM-L6-v2 model

# ==============================================
# 5. CSS THEME
# ==============================================
VINTAGE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Playfair+Display:wght@700&display=swap');
body, .gradio-container { background-color: #2b2118 !important; color: #e3d5c6 !important; font-family: 'Crimson Text', serif !important; }
.vintage-title { font-family: 'Playfair Display', serif; font-size: 3em; text-align: center; color: #d4af37; margin-bottom: 0.2em; text-shadow: 2px 2px 4px #000; }
.vintage-subtitle { text-align: center; font-size: 1.2em; font-style: italic; opacity: 0.8; margin-bottom: 2em; border-bottom: 1px solid #d4af37; padding-bottom: 1em; }
label span { color: #e3d5c6 !important; font-size: 1.1em !important; }
.block.gradio-checkboxgroup, .block.gradio-slider, .block.gradio-dropdown { background: #1f1812 !important; border: 1px solid #5c4033 !important; border-radius: 8px; padding: 15px; }
.vintage-button { background-color: #5c4033 !important; color: #fff !important; border: 1px solid #d4af37 !important; font-family: 'Playfair Display', serif !important; font-size: 1.2em !important; transition: all 0.3s ease; }
.vintage-button:hover { background-color: #d4af37 !important; color: #1f1812 !important; box-shadow: 0 0 10px #d4af37; }
.vintage-section-title { font-family: 'Playfair Display', serif; font-size: 1.8em; margin-top: 30px; margin-bottom: 15px; color: #d4af37; border-left: 4px solid #d4af37; padding-left: 10px; }
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #1f1812; }
::-webkit-scrollbar-thumb { background: #5c4033; }
"""

# ==============================================
# 6. MEMORY SYSTEM (with embedding caching)
# ==============================================
class SimpleMemory:
    def __init__(self, path="booknerd_memory.json"):
        self.path = path
        self.data = {"favorites": [], "save_later": [], "cache": {}}
        if not os.path.exists(self.path):
            self.save()
        self.load()

    def load(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)
        if "cache" not in self.data:
            self.data["cache"] = {}
            self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def add(self, key, entry):
        self.data.setdefault(key, [])
        # Prevent duplicates based on ID
        if not any(e["id"] == entry["id"] for e in self.data[key]):
            self.data[key].append(entry)
            self.save()

    def get(self, key):
        return self.data.get(key, [])

    def get_cached_embedding(self, book_id):
        """Retrieve cached embedding for a book if it exists."""
        return self.data["cache"].get(book_id)

    def cache_embeddings(self, book_id_to_embedding: Dict[str, list]):
        """Store multiple embeddings at once and save to disk a single time."""
        self.data["cache"].update(book_id_to_embedding)
        self.save()


memory = SimpleMemory()

# ==============================================
# 7. AGENTS
# ==============================================
@dataclass
class UserProfile:
    genres: List[str]
    taste_vector: np.ndarray  # Single vector representing the user's overall taste


class ProfileAnalyzerAgent:
    def analyze(self, genres_list):
        # 1. Encode manually selected genres
        cleaned = [g.strip().lower() for g in genres_list] if genres_list else ["fiction"]
        genre_vec = np.mean(EMBED_MODEL.encode(cleaned, convert_to_numpy=True), axis=0)

        # 2. Fetch favorites and blend their embeddings
        favorites = memory.get("favorites")
        fav_vectors = [np.array(fav["embedding"]) for fav in favorites if "embedding" in fav]

        # 3. Blend: 50% genre selection, 50% favorite vibes (if favorites exist)
        if fav_vectors:
            fav_vec = np.mean(fav_vectors, axis=0)
            combined_vector = np.mean([genre_vec, fav_vec], axis=0)
        else:
            combined_vector = genre_vec

        # Normalize to avoid drift
        norm = np.linalg.norm(combined_vector)
        if norm > 0:
            combined_vector = combined_vector / norm

        return UserProfile(genres=cleaned, taste_vector=combined_vector)


class SearchDataAgent:
    BASE_URL = "https://www.googleapis.com/books/v1/volumes"

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key provided. Set GOOGLE_BOOKS_API_KEY environment "
                    "variable or pass the key to the constructor."
                )
        self.api_key = api_key

    def search(self, subject: str, max_results: int = 40) -> List[Dict[str, Any]]:
        query = f"subject:{subject}"
        params = {
            "q": query,
            "key": self.api_key,
            "maxResults": min(max_results, 40),
            "printType": "books",
            "orderBy": "relevance",
            "langRestrict": "en",
            "projection": "full",
        }

        max_retries = 3
        timeout = 10

        for attempt in range(max_retries):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=timeout)
                if not response.ok:
                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(1.5 ** attempt)
                        continue
                    return []

                data = response.json()
                if "error" in data:
                    return []

                items = data.get("items", [])
                results = []
                for item in items:
                    info = item.get("volumeInfo", {})

                    book_id = item.get("id")
                    cached_embedding = memory.get_cached_embedding(book_id)

                    title = info.get("title", "Untitled")
                    authors = info.get("authors", ["Unknown"])
                    categories = info.get("categories", [])
                    description = info.get("description", "No description available.")

                    pub_date = info.get("publishedDate", "")
                    year = None
                    if pub_date:
                        match = re.search(r'\b(19|20)\d{2}\b', pub_date)
                        if match:
                            year = int(match.group())

                    images = info.get("imageLinks", {})
                    cover = images.get("thumbnail") or images.get("smallThumbnail")
                    if cover:
                        cover = cover.replace("http://", "https://")
                    else:
                        cover = "https://via.placeholder.com/300x450/2b2118/e3d5c6?text=No+Cover"

                    results.append({
                        "id": book_id,
                        "title": title,
                        "authors": authors,
                        "subjects": categories,
                        "cover": cover,
                        "year": year,
                        "description": description,
                        "_cached_embedding": cached_embedding,  # internal use
                    })
                return results

            except Exception:
                time.sleep(1.5 ** attempt)
        return []


class CriticAgent:
    def score(self, profile: UserProfile, books: List[Dict[str, Any]], surprise: bool = False):
        if not books:
            return []

        books_to_encode = []
        for b in books:
            if b.get("_cached_embedding"):
                b["_embedding"] = np.array(b["_cached_embedding"])
            else:
                books_to_encode.append(b)

        # Encode only books that weren't already cached
        if books_to_encode:
            corpora = []
            for b in books_to_encode:
                title = b.get("title", "")
                categories = " ".join(b.get("subjects", []))
                desc = b.get("description", "")
                corpus = f"{title} {categories} {desc}".strip() or title
                corpora.append(corpus)

            new_embeddings = EMBED_MODEL.encode(corpora, convert_to_numpy=True)

            new_cache_entries = {}
            for i, b in enumerate(books_to_encode):
                emb_list = new_embeddings[i].tolist()
                new_cache_entries[b["id"]] = emb_list
                b["_embedding"] = new_embeddings[i]

            # Single batched write instead of one write per book
            memory.cache_embeddings(new_cache_entries)

        all_embeddings = np.array([b.get("_embedding", np.zeros(EMBED_DIM)) for b in books])
        user_vec = profile.taste_vector.reshape(1, -1)
        similarities = cosine_similarity(user_vec, all_embeddings)[0]

        results = []
        for i, b in enumerate(books):
            # Cosine similarity ranges [-1, 1]; clamp to [0, 100] so a poor
            # match never renders as a negative percentage in the UI.
            score = max(0.0, min(similarities[i] * 100, 100.0))
            if b.get("year"):
                score = min(score + 2, 100.0)
            results.append({"book": b, "score": round(score, 2)})

        reverse_sort = not surprise
        return sorted(results, key=lambda x: x["score"], reverse=reverse_sort)


class Orchestrator:
    def __init__(self, profile_agent, search_agent, critic_agent, memory):
        self.profile_agent = profile_agent
        self.search_agent = search_agent
        self.critic_agent = critic_agent
        self.memory = memory

    def recommend(self, genres, k=5, year_range=(1800, 2024), surprise=False):
        profile = self.profile_agent.analyze(genres)
        all_books = []
        min_year, max_year = year_range

        for genre in profile.genres:
            found = self.search_agent.search(genre)
            all_books.extend(found)

        unique_books = {b["id"]: b for b in all_books}.values()
        filtered_books = [
            b for b in unique_books
            if b.get("year") and (min_year <= b["year"] <= max_year)
        ]

        scored = self.critic_agent.score(profile, filtered_books, surprise)
        top = scored[:k]

        results = []
        for s in top:
            b = s["book"]
            results.append({
                **b,
                "score": s["score"],
                "explanation": (
                    f"Match: {s['score']:.1f}%" if not surprise
                    else f"Exploration: {s['score']:.1f}% (stepping outside your bubble!)"
                ),
            })

        global LAST_RECOMMENDATIONS
        LAST_RECOMMENDATIONS = results
        return results


profile_agent = ProfileAnalyzerAgent()
search_agent = SearchDataAgent()
critic_agent = CriticAgent()
orch = Orchestrator(profile_agent, search_agent, critic_agent, memory)

LAST_RECOMMENDATIONS = []

# ==============================================
# 8. UI HELPERS
# ==============================================
def format_cards(books):
    if not books:
        return "<p style='color:#d4af37; text-align:center;'>No volumes found.</p>"
    html = "<div style='display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;'>"
    for b in books:
        title = b.get("title")
        authors = ", ".join(b.get("authors", []))
        year = b.get("year") or "N/A"
        cover = b.get("cover")
        score = b.get("score", 0)
        html += f"""
        <div style="width: 250px; background: #1f1812; border: 1px solid #5c4033; border-radius: 8px; padding: 15px; box-shadow: 5px 5px 15px rgba(0,0,0,0.5); display: flex; flex-direction: column;">
            <div style="height: 350px; overflow: hidden; border-radius: 4px; margin-bottom: 10px;">
                <img src="{cover}" style="width: 100%; height: 100%; object-fit: cover;">
            </div>
            <h3 style="margin: 5px 0; color: #d4af37; font-size: 1.1em; line-height: 1.2;">{title}</h3>
            <p style="font-size: 0.9em; opacity: 0.9; margin: 0;"><i>by {authors}</i></p>
            <p style="font-size: 0.85em; color: #8b5a2b; margin-top: 5px;">Published: {year}</p>
            <p style="font-size: 0.8em; color: #d4af37; margin-top: 5px;">🧠 Match: {score:.1f}%</p>
        </div>
        """
    html += "</div>"
    return html


def render_saved_list(key):
    items = memory.get(key)
    if not items:
        return "<p style='color:#777; font-style:italic;'>The shelf is empty.</p>"
    html = "<div style='display:flex; flex-wrap:wrap; gap:15px;'>"
    for b in items:
        html += f"""
        <div style="width:180px; padding:10px; background:#221e1a; border:1px solid #3e2f26; border-radius:6px;">
            <img src="{b['cover']}" style="width:100%; height:200px; object-fit:cover; border-radius:4px;">
            <h4 style="font-size:14px; color:#d4af37; margin:5px 0;">{b['title']}</h4>
        </div>
        """
    html += "</div>"
    return html


# ==============================================
# 9. GRADIO UI
# ==============================================
GENRE_OPTIONS = [
    "Fiction", "Mystery", "Thriller", "Romance", "Fantasy",
    "Science Fiction", "Historical Fiction", "Horror", "Classics",
    "Philosophy", "Psychology", "History", "Poetry"
]


def recommend_ui(genres, quantity, min_year, max_year, surprise):
    if not genres:
        gr.Info("Please select at least one genre.")
        return None, gr.update(choices=[], value=None)
    books = orch.recommend(genres, quantity, year_range=(min_year, max_year), surprise=surprise)
    if not books:
        gr.Info("No books found.")
        return format_cards([]), gr.update(choices=[], value=None)
    labels = [f"{b['title']} ({b.get('year', 'N/A')})" for b in books]
    return format_cards(books), gr.update(choices=labels, value=labels[0] if labels else None)


def add_fav_ui(selected_label):
    if not selected_label:
        gr.Info("No book selected.")
        return
    for b in LAST_RECOMMENDATIONS:
        label = f"{b['title']} ({b.get('year', 'N/A')})"
        if label == selected_label:
            # Generate embedding for the favorite so the AI can learn from it
            corpus = f"{b['title']} {' '.join(b.get('subjects', []))} {b.get('description', '')}"
            embedding = EMBED_MODEL.encode(corpus, convert_to_numpy=True).tolist()
            b_copy = b.copy()
            b_copy["embedding"] = embedding
            memory.add("favorites", b_copy)
            gr.Info(f"❤️ Saved '{b['title']}' - Your taste profile will adapt!")
            return
    gr.Info("Error: Book data not found.")


def add_save_ui(selected_label):
    if not selected_label:
        gr.Info("No book selected.")
        return
    for b in LAST_RECOMMENDATIONS:
        label = f"{b['title']} ({b.get('year', 'N/A')})"
        if label == selected_label:
            b_copy = b.copy()
            b_copy.pop("_embedding", None)
            b_copy.pop("_cached_embedding", None)
            memory.add("save_later", b_copy)
            gr.Info(f"🔖 Saved '{b['title']}' for later.")
            return
    gr.Info("Error: Book data not found.")


with gr.Blocks(css=VINTAGE_CSS, theme=gr.themes.Default(primary_hue="yellow")) as demo:
    gr.HTML("<div class='vintage-title'>🕯️ The BookNerd's Sanctuary</div>")
    gr.HTML("<div class='vintage-subtitle'>AI that learns from your favorites.</div>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Define Your Tastes")
            genres = gr.CheckboxGroup(GENRE_OPTIONS, label="Select Genres")

            gr.Markdown("### 2. Time Period")
            min_year_slider = gr.Slider(minimum=1800, maximum=2025, value=1900, step=1, label="Min Year")
            max_year_slider = gr.Slider(minimum=1800, maximum=2025, value=2024, step=1, label="Max Year")
            qty = gr.Slider(1, 10, value=4, step=1, label="Stack Size")

            surprise_mode = gr.Checkbox(label="🎲 Surprise Me (Explore outside your bubble)", value=False)

            btn = gr.Button("🔍 Consult the Archives", elem_classes="vintage-button")

            gr.Markdown("---")
            gr.Markdown("### 3. Actions")
            book_selector = gr.Dropdown([], label="Select Book from Results")
            with gr.Row():
                fav_btn = gr.Button("❤️ Add to Favorites", elem_classes="vintage-button")
                save_btn = gr.Button("🔖 Save for Later", elem_classes="vintage-button")

        with gr.Column(scale=2):
            gr.Markdown("### Recommended Volumes")
            out_display = gr.HTML()
            gr.Markdown("---")
            with gr.Tabs():
                with gr.TabItem("⭐ Favorites"):
                    fav_view = gr.HTML(render_saved_list("favorites"))
                    refresh_fav = gr.Button("Refresh Shelf")
                with gr.TabItem("📚 To Read Later"):
                    save_view = gr.HTML(render_saved_list("save_later"))
                    refresh_save = gr.Button("Refresh Shelf")

    btn.click(recommend_ui, [genres, qty, min_year_slider, max_year_slider, surprise_mode], [out_display, book_selector])
    fav_btn.click(add_fav_ui, book_selector, None)
    save_btn.click(add_save_ui, book_selector, None)
    refresh_fav.click(lambda: render_saved_list("favorites"), None, fav_view)
    refresh_save.click(lambda: render_saved_list("save_later"), None, save_view)

if __name__ == "__main__":
    demo.launch(debug=True)
