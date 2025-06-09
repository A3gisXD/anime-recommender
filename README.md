🎌 AI-Powered Manga & Anime Recommender
This is an interactive web app that helps users discover new anime and manga titles using artificial intelligence. Just select a series you like, and the system will instantly recommend similar titles based on genre, plot summaries, and more.

Built with Python, scikit-learn, and Streamlit, the app uses TF-IDF vectorization and cosine similarity to find and rank recommendations from a curated dataset.

🌟 Features
🔍 Smart Recommendations: Get AI-generated manga/anime suggestions based on content similarity.

📊 Optional Ratings Sorting: (Coming Soon) Boost recommendations using user ratings.

🎨 Visual Covers: (Coming Soon) Display anime covers for a more immersive experience.

🧠 Session Memory: (Coming Soon) Personalize suggestions based on your previous choices.

🚀 How to Run Locally
Clone the repo:
git clone https://github.com/YOUR_USERNAME/anime-recommender.git
cd anime-recommender
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run anime_recommender_app.py

📦 Dataset
The recommender is powered by a cleaned_dataset.csv file containing:

Title

Genre

Background summary

Thanks:
MyAnimeList 
Azathoth 

