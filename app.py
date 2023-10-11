import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your dataset
rp = pd.read_csv("C:\\Users\\chira\\OneDrive\\Desktop\\Cleaned_Indian_Food_Dataset.csv")


# Define a CountVectorizer
cv = CountVectorizer(max_features=500)
vectors = cv.fit_transform(rp['Cleaned-Ingredients']).toarray()
similarity = cosine_similarity(vectors)


def recommend(recipe):
    index = rp[rp['TranslatedRecipeName'] == recipe].index[0]
    distances = sorted(enumerate(similarity[index]), reverse=True, key=lambda x: x[1])
    recommended_recipes = [rp.iloc[i[0]]['TranslatedRecipeName'] for i in distances[1:11]]
    return recommended_recipes


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        user_input = request.form.get("ingredients")
        user_input = user_input.split(',')
        user_input_vector = cv.transform([' '.join(user_input)]).toarray()

        # Get cosine similarities between user input and recipes
        user_similarities = cosine_similarity(user_input_vector, vectors)

        # Find the top N recipe indices
        top_indices = user_similarities.argsort()[0, ::-1][:10]

        # Get recommended recipe names
        recommendations = [rp['TranslatedRecipeName'].iloc[idx] for idx in top_indices]

    return render_template("index.html", recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
