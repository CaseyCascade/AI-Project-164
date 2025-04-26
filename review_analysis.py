# Dataset basis
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tools
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#============# Prepare Data #============#
# Load and shuffle the dataset
dataset = load_dataset("Kwaai/IMDB_Sentiment", split="train").shuffle(seed=42)

# Show a sample text
print(dataset[0]["text"])

# Select a smaller subset for faster training (optional)
dataset = dataset.select(range(1000))

# Extract text and labels
texts = dataset["text"]
labels = dataset["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#========================================#

#================# KNN #=================#
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K Nearest Neighbors' Accuracy: {accuracy_knn:.2f}")
#========================================#

#============# Naive Bayes #=============#
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes' Accuracy: {accuracy_nb:.4f}")
#========================================#

#=========# Logistic Regression #========#
logreg_classifier = LogisticRegression(max_iter=1000)
logreg_classifier.fit(X_train, y_train)
y_pred_logreg = logreg_classifier.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression's Accuracy: {accuracy_logreg:.4f}")
#========================================#

#===========# Testing Section #==========#
custom_texts = [
    "This movie was absolutely fantastic and very enjoyable!",
    "I wish it was shorter."
]

# Transform custom texts using the same vectorizer
custom_X = vectorizer.transform(custom_texts)

# Predict using your best model (replace with the one you prefer)
custom_preds_logreg = logreg_classifier.predict(custom_X)

# Show predictions
for text, pred in zip(custom_texts, custom_preds_logreg):
    label = "Positive" if pred == 1 else "Negative"
    print(f"Text: \"{text}\"\nPrediction: {label}\n")
#========================================#