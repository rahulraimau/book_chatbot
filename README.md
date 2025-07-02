This project implements a simple content-based book recommendation chatbot. It leverages TF-IDF (Term Frequency-Inverse Document Frequency) for text representation and Cosine Similarity for calculating the similarity between books. The chatbot provides recommendations based on a user-provided book title, displaying the recommended book's title, description, and average rating. It also includes a basic analysis of the genre distribution among the recommended books.

Key Features:

Content-Based Recommendations: Recommends books based on the textual similarity of their descriptions and titles.

TF-IDF Vectorization: Converts text data into numerical features for similarity calculation.

Cosine Similarity: Measures the similarity between book features.

Interactive Chatbot: A command-line interface for users to request recommendations.

Detailed Output: Provides title, description, and average rating for each recommended book.

Genre Distribution Analysis: Shows the breakdown of categories/genres for the recommended books.

How to Run:

1.Ensure you have Python installed.

2.Install required libraries: pandas, scikit-learn.

3.Place data.csv in the same directory as the Python script.

4.Run the script: python your_script_name.py

5.Follow the prompts in the console to get recommendations.
