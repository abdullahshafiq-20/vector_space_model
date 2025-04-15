# Vector Space Model Information Retrieval System

This project implements a Vector Space Model (VSM) Information Retrieval System that supports:
- Ranked retrieval using TF-IDF weighting
- Cosine similarity scoring
- Query processing with tokenization, stemming, and stop-word removal

---

## Technologies Used

- **Python** – Core programming language used for development.
- **NLTK** – Library for natural language processing (tokenizing, stemming, stop-word removal, etc.).
- **NumPy** – For efficient vector and matrix operations.
- **Streamlit** – Framework for creating an interactive web UI.

---

## Project Structure

- **vector_space_model.py** – Implements the core VSM retrieval logic.
- **main.py** – Streamlit-based web UI for user interaction.
- **requirements.txt** – Lists dependencies needed to run the project.
- **Abstracts/** – Contains text documents to be indexed.
- **stop_words.txt** – Stores common stop words used in text processing.
- **indexes.json** – Stores precomputed indexes for efficient retrieval.

---

## Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Console Version:**
   ```bash
   python vector_space_model.py
   ```

3. **Run the Web UI:**
   ```bash
   streamlit run main.py
   ```

---

## Query Examples

- **Simple Query:**
  ```
  computer science
  ```
- **Ranked Retrieval:**  
  The system returns documents ranked by their relevance to the query using cosine similarity.

---

## Application Links

- **Deployed App:** [Access the Web App](https://your-vsm-app-link.streamlit.app/)

---

## Additional Information

- **Indexing:** Processed documents are stored in `indexes.json` for faster access.
- **TF-IDF Weighting:** Uses term frequency-inverse document frequency for document and query vectors.
- **Extensibility:** Easily adaptable for additional retrieval functionalities.

This documentation provides an overview of the Vector Space Model Information Retrieval System. For further details, refer to the source code comments and inline documentation.