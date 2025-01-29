import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from nltk.util import bigrams, trigrams, ngrams

#Download NLTK data
nltk.download("punkt_tab")

# Step 1: Sample text corpus
corpus = """
Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.

ML finds application in many fields, including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine. The application of ML to business problems is known as predictive analytics.

Statistics and mathematical optimization (mathematical programming) methods comprise the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) via unsupervised learning.

From a theoretical viewpoint, probably approximately correct learning provides a framework for describing machine learning.
"""

# Step 2: Remove unneccecary special characters and tokenize the text
replacements = {"," : "", "." : "", "_" : "", "(": "", ")" : ""} # Remove special characters
for old, new in replacements.items():
    corpus = corpus.replace(old, new)
print(corpus)
tokens = word_tokenize(corpus.lower())
print(tokens , '\n')
print(len(tokens) , '\n')

# Step 3: Build a trigram model
trigram_model = defaultdict(Counter)
for w1, w2, w3 in trigrams(tokens):
    trigram_model[(w1, w2)][w3] += 1
# Step 4: Predict the next word
print(trigram_model[("machine", "learning")])

def predict_next_words(words):
    words = words.split()  # Convert input string into a list
    if len(words) < 2:
      return "Please enter at least two words"
    else:
      key = (words[-2], words[-1])  # Use only the last two words
      next_words = trigram_model.get(key, {})

    if next_words:
        return next_words.most_common(3)  # Get top 3 predictions
    else:
        return "No words found"

# Example usage
user_input = input("Enter two words: ").lower()
print(f"Predicted next words: {  predict_next_words(user_input) }")