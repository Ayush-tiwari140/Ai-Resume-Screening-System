import io
import re
import docx
import numpy as np
import pandas as pd
import PyPDF2
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data assets quietly
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)


class ResumeScreeningEngine:
    """AI Engine for extracting text, preprocessing via NLP, and ranking resumes."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    def extract_text(self, uploaded_file) -> str:
        """Extracts text directly from Streamlit uploaded file buffers."""
        file_name = uploaded_file.name.lower()
        text = ""

        try:
            if file_name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + " "

            elif file_name.endswith(".docx"):
                doc = docx.Document(uploaded_file)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + " "

            elif file_name.endswith(".txt"):
                text = uploaded_file.read().decode("utf-8", errors="ignore")

        except Exception as e:
            st.error(f"Error reading file '{uploaded_file.name}': {e}")

        return text.strip()

    def preprocess_text(self, raw_text: str) -> str:
        """Cleans, tokenizes, removes stopwords, and lemmatizes input text."""
        text = raw_text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        tokens = word_tokenize(text)
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]
        return " ".join(cleaned_tokens)

    def rank_resumes(self, job_description: str, uploaded_files: list) -> pd.DataFrame:
        """Ranks uploaded resume files against the provided job description."""
        clean_jd = self.preprocess_text(job_description)

        names, raw_texts, cleaned_corpus = [], [], []

        for file in uploaded_files:
            raw_text = self.extract_text(file)
            if not raw_text:
                st.warning(f"Could not extract text from file: {file.name}")
                continue

            cleaned_text = self.preprocess_text(raw_text)
            names.append(file.name)
            raw_texts.append(raw_text)
            cleaned_corpus.append(cleaned_text)

        if not cleaned_corpus:
            return pd.DataFrame()

        # Combine JD + Resumes into corpus for vectorization
        full_corpus = [clean_jd] + cleaned_corpus
        tfidf_matrix = self.vectorizer.fit_transform(full_corpus)

        jd_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]

        # Calculate Cosine Similarity
        similarity_scores = cosine_similarity(jd_vector, resume_vectors).flatten()

        results = []
        for name, score in zip(names, similarity_scores):
            match_percentage = round(score * 100, 2)

            if match_percentage >= 60.0:
                status = "Strong Match"
            elif match_percentage >= 35.0:
                status = "Potential Match"
            else:
                status = "Low Match"

            results.append({
                "Candidate Resume": name,
                "Match Score (%)": match_percentage,
                "Status": status,
            })

        # Format output DataFrame
        df_results = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)
        df_results.reset_index(drop=True, inplace=True)
        df_results.index += 1  # 1-based index ranking
        return df_results


# --- STREAMLIT UI LAYOUT ---
def main():
    st.set_page_config(page_title="AI Resume Screener", page_icon="", layout="wide")

    st.title("AI Resume Screening System")
    st.markdown(
        "Upload candidate resumes (**PDF, DOCX, TXT**) and paste a **Job Description** "
        "to calculate match scores and rank top candidates using NLP vector space analysis."
    )

    st.divider()

    # Create two layout columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Job Description")
        job_description = st.text_area(
            "Paste the job requirements here:",
            height=280,
            placeholder="e.g., We are looking for a Data Scientist skilled in Python, NLP, PyTorch, SQL...",
        )

    with col2:
        st.subheader("2. Candidate Resumes Upload")
        uploaded_files = st.file_uploader(
            "Upload multiple resumes:",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Select one or multiple files in PDF, DOCX, or TXT formats.",
        )

    st.divider()

    # Action Button
    if st.button("Screen & Rank Resumes", type="primary", use_container_width=True):
        if not job_description.strip():
            st.error("Please provide a Job Description before screening.")
            return

        if not uploaded_files:
            st.error("Please upload at least one candidate resume file.")
            return

        with st.spinner("Processing documents, extracting text, and computing NLP similarities..."):
            engine = ResumeScreeningEngine()
            results_df = engine.rank_resumes(job_description, uploaded_files)

        if results_df.empty:
            st.error("Failed to process uploaded resumes. Please verify file formats.")
            return

        # Display Summary Dashboard Metrics
        st.subheader("3. Screening Results Dashboard")

        top_candidate = results_df.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Resumes Analyzed", len(results_df))
        m2.metric("Top Candidate", top_candidate["Candidate Resume"])
        m3.metric("Highest Match Score", f"{top_candidate['Match Score (%)']}%")

        # Display Interactive Table
        st.dataframe(
            results_df,
            column_config={
                "Match Score (%)": st.column_config.ProgressColumn(
                    "Match Score (%)",
                    help="Cosine Similarity score represented as a percentage",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                )
            },
            use_container_width=True,
        )


if __name__ == "__main__":
    main()