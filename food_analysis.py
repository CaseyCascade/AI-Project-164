from datasets import load_dataset
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# load the food-101 dataset
dataset = load_dataset("ethz/food101", split="train[:1000]")

# format images
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# convert to feature matrix
X = np.stack([transform(img).view(-1).numpy() for img in dataset["image"]])

y = np.array(dataset["label"])

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize classifier
knn = KNeighborsClassifier(n_neighbors=5)

# fit on training data
knn.fit(X_train, y_train)

# predict on test data
y_pred = knn.predict(X_test)

# accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")

# test a custom iamge
custom_img = Image.open("hamburger.jpg").convert("RGB")
img_tensor = transform(custom_img).view(1, -1).numpy()

pred = knn.predict(img_tensor)
label = dataset.features["label"].int2str([pred[0]])[0]
print(f"Predicted: {label}")