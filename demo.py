from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tqdm as tqdm

# 1) Load and normalize EMNIST Balanced
print("Loading EMNIST...")
mnist = fetch_openml("EMNIST_Balanced", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Optional: Apply PCA to reduce dimensionality
USE_PCA = False
if USE_PCA:
    pca = PCA(n_components=100)  # Try 50–150
    X = pca.fit_transform(X)

# 2) Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# 3) Logistic Regression
# Accuracy: 0.69
def LR_test():
    lr = LogisticRegression(solver='saga', max_iter=100, tol=0.00127722364893473, C=1.6176755124355395)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))

# 4) KNN
# Accuracy: 0.78
def KNN_test():
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

if __name__ == "__main__":
    LR_test()
    KNN_test()