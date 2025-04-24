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
dataset = load_dataset("coallaoh/ImageNet-AB", split="train").shuffle(seed=42)

# Select the first N randomly sampled examples
dataset = dataset.select(range(1000))  # e.g., 1000 random samples

# format images
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# convert to feature matrix
X = np.stack([transform(img).view(-1).numpy() for img in dataset["image"]])
y = np.array(dataset["label"])

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#========================================#