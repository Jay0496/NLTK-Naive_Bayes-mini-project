import random
import nltk
from nltk.corpus import movie_reviews
from nltk import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Load the Movie Reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Calculate word frequencies
all_words = FreqDist(word.lower() for word in movie_reviews.words())
word_features = list(all_words.keys())[:2000]

# Define a feature extraction function
def document_features(document):
    document_words = set(document)
    return {word: (word in document_words) for word in word_features}

# Prepare feature sets
feature_sets = [(document_features(doc), category) for (doc, category) in documents]
train_set, test_set = feature_sets[100:], feature_sets[:100]

# Train the classifier
classifier = NaiveBayesClassifier.train(train_set)

# Test the classifier
print(f"Accuracy: {accuracy(classifier, test_set)}")

# Display the top 10 most informative features
classifier.show_most_informative_features(10)
