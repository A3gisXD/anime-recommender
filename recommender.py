import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df['combined'] = df['genre'] + " " + df['background']
    return df

# Build the TF-IDF matrix and cosine similarity
@st.cache_resource
def build_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return tfidf_matrix, cosine_sim, indices

# Get recommendations
def get_recommendations(title, df, cosine_sim, indices, top_n=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[rec_indices].tolist()

# Streamlit UI
st.title("🎌 AI-Powered Anime Recommender")
st.write("Pick an anime you like, and we'll recommend similar titles!")

df = load_data()
_, cosine_sim, indices = build_recommender(df)

selected_title = st.selectbox("Select an anime title:", df['title'].tolist())

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_title, df, cosine_sim, indices)
    if recommendations:
        st.subheader("💡 You might also like:")
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.warning("Sorry, no recommendations found.")
