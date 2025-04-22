from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

debug = False

# 1) Load and cast labels
mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# 3) Logistic Regression
lr = LogisticRegression(solver='saga', max_iter=100, tol=0.1)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))

# 4) KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# 5) Sample some KNN predictions
if debug:
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), size=5, replace=False)
    for i in sample_idx:
        print(f"Index {i}: True={y_test[i]}  Pred={y_pred_knn[i]}")
        plt.imshow(X_test[i].reshape(28,28), cmap='gray')
        plt.title(f"True={y_test[i]} Pred={y_pred_knn[i]}")
        plt.axis('off')
        plt.show()
