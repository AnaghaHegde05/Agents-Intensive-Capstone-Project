# The BookNerd’s Sanctuary: A Multi-Agent AI Book Recommendation System

> **A vintage-themed multi-agent AI system that recommends books using Google Books API, intelligent scoring, and long-term memory.**

## 📖 Project Description

### Problem Statement

Recommending books in a meaningful and personalized way is more complex than simply listing popular titles. Most digital recommendation systems rely on broad trends and popularity metrics, overlooking genre nuance, era preferences, and the personal nature of reading. What readers often want is the feeling of a curated discovery experience, similar to browsing an old library and stumbling upon something that truly matches their taste.

This challenge becomes even more interesting when recommendations must be generated dynamically using real-time, external book data rather than from fixed, preprocessed datasets. Solving this requires coordinated reasoning, structured filtering, metadata evaluation, and personalized memory—capabilities that extend beyond the scope of a single procedural script.

### Why Agents?

Agents are well suited for this task because the recommendation workflow naturally decomposes into specialized, interdependent reasoning steps. The system requires agents that can clean and interpret user preferences, interact with an external tool for real-time data retrieval, evaluate relevance, orchestrate decision flows, and maintain persistent user memory.

A multi-agent architecture supports:

  * Clear separation of responsibilities
  * Improved modularity and error isolation
  * Interpretable, stepwise decision-making
  * Scalable reasoning workflows
  * A design that mirrors how human librarians collaborate

Each agent performs one part of the reasoning chain, and their coordinated interaction leads to a cohesive, efficient recommendation experience. Instead of one monolithic block of logic, the pipeline behaves like a team of specialists working together.

-----

## 🏗️ Architecture Overview

The system consists of four sequentially orchestrated agents, each responsible for a specific stage of the recommendation pipeline.

### 1\. ProfileAnalyzerAgent

Processes genre selections from the user and converts them into a clean, normalized `UserProfile` object. This ensures consistent downstream processing regardless of input variability.

### 2\. SearchDataAgent (Tool-Agent)

Interacts directly with the **Google Books API**, retrieving live book metadata such as titles, authors, subjects, publication years, descriptions, and cover images. This agent handles missing or inconsistent metadata and formats results into normalized book objects.

### 3\. CriticAgent

Scores each candidate book based on its genre alignment and metadata completeness. It ranks results to ensure that the most relevant and well-described titles appear first.

### 4\. Orchestrator Agent

Acts as the central controller of the recommendation process. It coordinates the entire pipeline by:

  * Running the agents in sequence
  * Deduplicating books
  * Filtering by user-defined publication year range
  * Selecting the top K recommendations
  * Maintaining session-level state for user actions

### Long-Term Memory (SimpleMemory)

A JSON-backed memory module stores the user’s **Favorites** and **Save-for-Later** shelves across sessions. This provides continuity, allowing the system to behave like a librarian who remembers what you liked previously.

-----

## 🎨 User Interface

The front-end interface is built with **Gradio**, styled using a custom vintage library theme. The UI integrates:

  * Genre selectors
  * Year range sliders
  * A recommendation trigger
  * Dynamically rendered book cards
  * Persistent shelves for Favorites and Save-for-Later

The design mirrors the ambiance of an old reading room, giving the system both utility and aesthetic character.

-----

## 🚀 Demo: The Solution in Action

When a user interacts with the system, they select genres and a publication year range. Once they click **“Consult the Archives”**, the multi-agent pipeline activates:

1.  **Preferences are interpreted.**
2.  **The Google Books API is queried in real time.**
3.  **Results are cleaned, scored, and ranked.**
4.  **The top recommendations are displayed as vintage-style cards.**

Each card shows the book cover, title, authors, publication year, and supports actions such as:

  * Add to Favorites
  * Save for Later
<img width="1920" height="809" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/6edc1827-c6e8-426e-96e0-34975b1d9f31" />
<img width="1920" height="830" alt="Screenshot (54)" src="https://github.com/user-attachments/assets/b7ac4c6c-d7f5-4964-9745-90e6c08ddc8e" />
<img width="1920" height="812" alt="Screenshot (55)" src="https://github.com/user-attachments/assets/24b50dfc-a7be-4702-8ee3-5a69bf486cfc" />
<img width="1920" height="827" alt="Screenshot (56)" src="https://github.com/user-attachments/assets/73debb6f-c24e-43d6-a21b-36dcd7939436" />
<img width="1920" height="827" alt="Screenshot (57)" src="https://github.com/user-attachments/assets/f8f3763b-ecd2-449b-a0a6-87feaa609ef1" />
<img width="1920" height="811" alt="Screenshot (58)" src="https://github.com/user-attachments/assets/40b27eb0-a2d3-4798-8d19-8f0183e5051d" />

The Favorites and Save-for-Later shelves update instantly and persist across sessions. The result is a smooth, engaging discovery experience that feels like having an intelligent librarian curate books just for you.

-----

## 🛠️ The Build (Tools & Technologies)

### Core Tools

  * **Python**
  * **Gradio** (for the UI)
  * **Google Books API** (for real-time metadata)

### Agent Concepts Implemented

  * Multi-agent orchestration
  * Sequential reasoning
  * Context engineering
  * Stateful session management
  * Long-term persistent memory

### Key Libraries

  * `requests` (API communication)
  * `dataclasses` (Structured objects)
  * `SentenceTransformers` (Preloaded for future semantic upgrades)
  * `json` (Memory storage)

### Additional Components

  * A custom vintage-themed CSS style
  * HTML-based rendering for recommendation cards
  * Safe file handling for memory persistence

-----

## 🔮 Future Improvements

Several meaningful extensions could improve the system further:

  -  **Semantic Similarity Scoring:** Use embeddings to match books based on deeper thematic patterns.
  -  **Parallel Execution:** Enable parallel agent execution to fetch multiple genres at once, improving latency.
  -  **LLM-Guided Critic:** Develop an LLM-guided `CriticAgent` to explain *why* a book fits a user’s taste.
  -  **AI Summaries:** Generate custom short summaries for recommended books using LLMs.
  -  **Database Integration:** Move from JSON to a database such as SQLite or Supabase to support multi-user environments.
  -  **Cloud Deployment:** Deploy as a cloud-hosted application with login systems and personalized libraries.
  -  **Multi-User Sessions:** Introduce handling for classrooms, book clubs, or family members.
