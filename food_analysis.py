from datasets import load_dataset
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#============# Prepare Data #============#
# Load and shuffle the dataset
dataset = load_dataset("ethz/food101", split="train").shuffle(seed=42)

# Define a transform to resize and convert PIL images to NumPy arrays
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize for faster processing
    transforms.ToTensor()
])

# Apply the transform and extract image arrays and labels
images = []
labels = []
label_to_idx = {label: idx for idx, label in enumerate(set(dataset['label']))}

for item in dataset.select(range(1000)):  # Limit for performance
    image = item['image'].convert("RGB")  # Already a PIL Image
    image = transform(image)
    images.append(image.numpy().flatten())
    labels.append(item['label'])  # Already an integer label

X = np.array(images)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#========================================#

#================# KNN #=================#
# initialize classifier
knn = KNeighborsClassifier(n_neighbors=5)

# fit on training data
knn.fit(X_train, y_train)

# predict on test data
y_pred_knn = knn.predict(X_test)

# accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K Nearest Neighbors' Accuracy: {accuracy_knn:.2f}")
#========================================#

#============# Naive Bayes #=============#
# initialize the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# train the classifier
nb_classifier.fit(X_train, y_train)

# predict on the test set
y_pred_nb = nb_classifier.predict(X_test)

# evaluate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes' Accuracy: {accuracy_nb:.4f}")
#========================================#

#=========# Logistic Regression #========#
# initialize Logistic Regression
logreg_classifier = LogisticRegression(max_iter=1000, solver='lbfgs')

# train the classifier
logreg_classifier.fit(X_train, y_train)

# predict on the test set
y_pred_logreg = logreg_classifier.predict(X_test)

# evaluate accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression's Accuracy: {accuracy_logreg:.4f}")
#========================================#

#===========# Testing Section #==========#
#========================================#