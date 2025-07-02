import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Data Import and Initial Exploration (from previous steps)
# Load the data - assuming preprocessed_data.csv is available
# We need the original DataFrame to get descriptions and ratings, so let's load it again.
df_original = pd.read_csv(r"C:\Users\DELL\Downloads\data.csv (1)\data.csv")
df_text = pd.read_csv(r"C:\Users\DELL\PycharmProjects\PythonProject7\preprocessed_data.xls")

# Merge df_text with df_original to get descriptions and ratings
# Assuming 'title' is a common column and unique enough for merging, or use 'isbn13' if available in preprocessed_data.csv
# For simplicity, let's assume `df_text` was derived from `df_original` and their indices align,
# or we can re-merge based on title if `preprocessed_data.csv` doesn't contain `isbn13`.
# For robust merging, it's better to ensure `preprocessed_data.csv` retains a unique identifier like `isbn13`.
# For this update, I'll assume the original `df` is loaded and can be used directly for descriptions and ratings lookup.
# If `preprocessed_data.csv` does not have `isbn13`, we need to re-think how to link.
# Let's use the original `df` for all lookups, assuming its index matches `tfidf_matrix` rows.

# Fill missing values in `description` and `title` columns for the original df as well,
# in case it's used for direct lookup in get_recommendations
df_original['description'] = df_original['description'].fillna('No description available.')
df_original['average_rating'] = df_original['average_rating'].fillna('N/A')

# 3. Model Training (TF-IDF and Cosine Similarity)
# Initialize the TF-IDF Vectorizer
# stop_words='english' removes common English words that don't add much meaning
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the 'combined_features' column
# This converts the text data into a matrix of TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(df_text['combined_features'])

# Compute the cosine similarity matrix
# Cosine similarity measures the similarity between two non-zero vectors
# It is a commonly used metric for text similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# 4. Chatbot Prediction Function
def get_recommendations(title, cosine_sim_matrix=cosine_sim, df_titles_features=df_text, df_full_data=df_original):
    """
    Generates book recommendations based on a given book title using cosine similarity,
    including book description and average rating.

    Args:
        title (str): The title of the book for which to find recommendations.
        cosine_sim_matrix (np.array): The pre-computed cosine similarity matrix.
        df_titles_features (pd.DataFrame): DataFrame with 'title' and 'combined_features' for TF-IDF.
        df_full_data (pd.DataFrame): The original DataFrame containing all book details (description, rating).

    Returns:
        list: A list of dictionaries, each containing 'title', 'description', and 'average_rating'
              for recommended books, or a string message if the book is not found.
    """
    # Get the index of the book that matches the title (case-insensitive)
    idx = df_titles_features[df_titles_features['title'].str.contains(title, case=False, na=False)].index

    if len(idx) == 0:
        return "Book not found. Please try another title."
    else:
        # If multiple matches, take the first one (can be improved for user selection)
        idx = idx[0]

    # Get the pairwise similarity scores of all books with the selected book
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort the books based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar books. Exclude the book itself.
    sim_scores = sim_scores[1:6]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Retrieve the titles, descriptions, and average ratings for the recommended books
    recommended_books_data = []
    for i in book_indices:
        book_title = df_full_data['title'].iloc[i]
        book_description = df_full_data['description'].iloc[i]
        book_rating = df_full_data['average_rating'].iloc[i]
        recommended_books_data.append({
            'title': book_title,
            'description': book_description,
            'average_rating': book_rating
        })

    return recommended_books_data


# 5. Chatbot Interaction
print("Welcome to the Book Recommendation Chatbot!")
print("Enter a book title to get recommendations, or type 'exit' to quit.")

while True:
    user_input = input("\nEnter a book title: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    recommendations = get_recommendations(user_input)

    if isinstance(recommendations, list):
        print("\nRecommended books:")
        for book in recommendations:
            print(f"- Title: {book['title']}")
            print(f"  Description: {book['description']}")
            print(f"  Average Rating: {book['average_rating']}\n")
    else:
        print(recommendations)