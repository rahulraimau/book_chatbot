import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- 1. Load Data ---
@st.cache_data
def load_data():
    df_original = pd.read_csv(r"C:\Users\DELL\Downloads\data.csv (1)\data.csv")
    df_text = pd.read_csv(r"C:\Users\DELL\PycharmProjects\PythonProject7\preprocessed_data.xls")
    df_original['description'] = df_original['description'].fillna('No description available.')
    df_original['average_rating'] = df_original['average_rating'].fillna('N/A')
    return df_original, df_text


df_original, df_text = load_data()


# --- 2. TF-IDF Vectorization ---
@st.cache_resource
def compute_similarity_matrix(text_df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_df['combined_features'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


cosine_sim = compute_similarity_matrix(df_text)


# --- 3. Recommendation Function ---
def get_recommendations(title, cosine_sim_matrix=cosine_sim, df_titles_features=df_text, df_full_data=df_original):
    idx = df_titles_features[df_titles_features['title'].str.contains(title, case=False, na=False)].index
    if len(idx) == 0:
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    book_indices = [i[0] for i in sim_scores]
    recommended_books = df_full_data.iloc[book_indices][['title', 'description', 'average_rating']]
    return recommended_books


# --- 4. Streamlit UI ---
st.set_page_config(page_title="📚 Book Recommender", layout="centered")
st.title("📖 Book Recommendation Chatbot")
st.markdown("Enter a book title to get top 5 similar book recommendations based on TF-IDF.")

user_input = st.text_input("Enter a book title:", "")

if user_input:
    with st.spinner("Finding recommendations..."):
        recommendations = get_recommendations(user_input)

    if recommendations is not None:
        st.success("Here are your recommendations:")
        for _, row in recommendations.iterrows():
            st.subheader(row['title'])
            st.write(f"**Rating:** {row['average_rating']}")
            st.write(f"**Description:** {row['description']}")
            st.markdown("---")
    else:
        st.error("❌ Book not found. Please try another title.")

st.sidebar.markdown("Made with ❤️ using Streamlit")