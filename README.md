# Movie Reviews Sentiment Analysis Project

## **Project Overview**
This project implements a sentiment analysis system using the NLTK library. By leveraging the `movie_reviews` dataset, we classify movie reviews into two categories: *positive* and *negative*. The system is built using the Naive Bayes classifier and demonstrates the basics of natural language processing (NLP) and machine learning.

---

## **Why This Project Was Created**
1. **Learn Text Classification**: To understand the process of building a basic text classifier using a popular NLP library.
2. **Explore Natural Language Processing**: To gain experience with preprocessing text data and extracting meaningful features.
3. **Understand Naive Bayes Classifier**: To learn how probabilistic classifiers work and their applications in real-world problems.
4. **Apply NLP Concepts to Other Projects**: To use this knowledge in other domains requiring text analysis, such as customer feedback or social media sentiment analysis.

---

## **What I Learned**
1. **Text Preprocessing**:
   - Tokenizing text into words.
   - Filtering out stopwords and irrelevant terms.
   - Extracting features from text data for classification.
2. **Feature Engineering**:
   - Selecting the most relevant words from the dataset for accurate classification.
3. **Machine Learning Basics**:
   - Training and testing a Naive Bayes classifier.
   - Evaluating the classifier's performance using accuracy metrics.
4. **NLTK Toolkit**:
   - Working with NLTK's prebuilt corpora.
   - Using functions like `FreqDist` and `stopwords` for efficient preprocessing.
5. **Interpreting Outputs**:
   - Understanding the most informative features that influence classification.

---

## **How the Code Works**

### 1. **Dataset Loading**
The `movie_reviews` dataset is loaded from NLTK, where each review is categorized as either *positive* or *negative*. Reviews are tokenized into words.

```python
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
```
- **`movie_reviews.words(fileid)`**: Returns a list of words in a review.
- **Shuffling**: Ensures that the dataset is randomly split into training and testing sets.

### 2. **Feature Extraction**
The most common words in the dataset are selected as features. For each document, a dictionary is created indicating the presence or absence of these words.

```python
from nltk import FreqDist
all_words = FreqDist(word.lower() for word in movie_reviews.words())
word_features = list(all_words.keys())[:2000]  # Top 2000 words

def document_features(document):
    return {word: (word in document) for word in word_features}
```
- **`FreqDist`**: Counts the frequency of each word.
- **Feature Dictionary**: Maps each word to a Boolean indicating its presence in a review.

### 3. **Train-Test Split**
The dataset is split into training and testing sets, using 90% for training and 10% for testing.

```python
feature_sets = [(document_features(doc), category) for (doc, category) in documents]
train_set, test_set = feature_sets[100:], feature_sets[:100]
```

### 4. **Training the Classifier**
The Naive Bayes classifier is trained using the training set.

```python
from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_set)
```
- **Naive Bayes**: A probabilistic model that assumes features are independent given the class label.

### 5. **Evaluation**
The classifier's accuracy is tested on the unseen test set, and the most informative features are displayed.

```python
from nltk.classify.util import accuracy
print(f"Accuracy: {accuracy(classifier, test_set)}")
classifier.show_most_informative_features(10)
```
- **Accuracy**: Measures the proportion of correctly classified reviews.
- **Most Informative Features**: Lists words with the highest contribution to distinguishing between classes.

---

## **Output Explanation**
1. **Accuracy**: Displays the performance of the model on the test set. For example:
   ```
   Accuracy: 0.85
   ```
   This means 85% of the test reviews were correctly classified.

2. **Most Informative Features**:
   - Example Output:
     ```
     Most Informative Features
          contains(horrible) = True           neg : pos    =     15.0 : 1.0
          contains(outstand) = True           pos : neg    =     12.3 : 1.0
     ```
   - Interpretation:
     - Words like *"horrible"* strongly indicate a negative review.
     - Words like *"outstanding"* strongly indicate a positive review.

---

## **How to Run the Project**
1. Install the required library:
   ```bash
   pip install nltk
   ```
2. Download the `movie_reviews` dataset:
   ```python
   import nltk
   nltk.download('movie_reviews')
   ```
3. Run the script in your Python environment.
4. Observe the accuracy and the most informative features printed in the output.

---

## **Future Applications**
This project provides a foundation for applying text classification to:
- Customer feedback analysis.
- Social media sentiment analysis.
- Email spam detection.

---

### **Potential Extensions**
1. Use a larger feature set for improved accuracy.
2. Experiment with different classifiers like SVM or Decision Trees.
3. Apply the workflow to other datasets or domains.

---
