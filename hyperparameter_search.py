from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.stats import loguniform, randint
from contextlib import contextmanager
from datasets import load_dataset
import joblib
import warnings
import os
import tqdm
import json


# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# TQDM joblib integration
class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
    def __init__(self, *args, tqdm_obj=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm = tqdm_obj

    def __call__(self, *args, **kwargs):
        self.tqdm.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)

@contextmanager
def tqdm_joblib(tqdm_object):
    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = lambda *args, **kwargs: TqdmBatchCallback(*args, tqdm_obj=tqdm_object, **kwargs)
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback

# TODO Search Parameters
n_iterations = 50
cv_folds = 5
max_iter = 300

def review_search():
    #============# Prepare Data #============#
    # Load and shuffle the dataset
    dataset = load_dataset("Kwaai/IMDB_Sentiment", split="train").shuffle(seed=42)

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

    knn_results = KNN(X_train, X_test, y_train, y_test)
    nb_results = NB(X_train, X_test, y_train, y_test)
    lr_results = LR(X_train, X_test, y_train, y_test)

    combined_results = {
    "Logistic Regression": lr_results,
    "Naive Bayes": nb_results,
    "KNN": knn_results
    }

    output_dir = "Hyperparameter Search"
    os.makedirs(output_dir, exist_ok=True)

    # 4) Save to JSON
    output_path = os.path.join(output_dir, "review_hyperparameters.json")
    with open(output_path, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"✅ Saved combined results to {output_path}")

'''
def EMNIST_search():
        emnist = fetch_openml("EMNIST_Balanced", version=1, as_frame=False)
        X, y = emnist.data, emnist.target.astype(int)
        X = X / 255.0

        # Optional: Apply PCA to reduce dimensionality
        USE_PCA = True
        if USE_PCA:
            pca = PCA(n_components=400)
            X = pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        # Then subsample training set
        train_subsample_size = int(0.7 * len(X_train))  # use 30% of training data
        X_train = X_train[:train_subsample_size]
        y_train = y_train[:train_subsample_size]

        lr_results = LR(X_train, X_test, y_train, y_test)
        knn_results = KNN(X_train, X_test, y_train, y_test)

        combined_results = {
        "Logistic Regression": lr_results,
        "KNN": knn_results
        }

        output_dir = "Hyperparameter Search"
        os.makedirs(output_dir, exist_ok=True)

        # 4) Save to JSON
        output_path = os.path.join(output_dir, "EMNIST_hyperparameters.json")
        with open(output_path, "w") as f:
            json.dump(combined_results, f, indent=2)

        print(f"✅ Saved combined results to {output_path}")

# Load and preprocess MNIST
def MNIST_search():
    mnist = fetch_openml("mnist_784", as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)\
    
    lr_results = LR(X_train, X_test, y_train, y_test)
    knn_results = KNN(X_train, X_test, y_train, y_test)

    combined_results = {
    "Logistic Regression": lr_results,
    "KNN": knn_results
    }

    output_dir = "Hyperparameter Search"
    os.makedirs(output_dir, exist_ok=True)

    # 4) Save to JSON
    output_path = os.path.join(output_dir, "MNIST_hyperparameters.json")
    with open(output_path, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"✅ Saved combined results to {output_path}")
'''

def KNN(X_train, X_test, y_train, y_test):
    # --- Randomized Search for KNN ---
    knn_pipeline = Pipeline([
        ('classifier', KNeighborsClassifier())
    ])

    knn_param_dist = {
        'classifier__n_neighbors': randint(3, 8),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]  # 1=Manhattan, 2=Euclidean
    }

    knn_search = RandomizedSearchCV(
        knn_pipeline,
        param_distributions=knn_param_dist,
        n_iter=n_iterations,
        cv=cv_folds,
        n_jobs=-1
    )

    print("\nFitting KNN with RandomizedSearchCV...")
    with tqdm_joblib(tqdm.tqdm(total=n_iterations * cv_folds, desc="KNN Search Progress")):
        knn_search.fit(X_train, y_train)

    print("Best KNN Accuracy:", knn_search.score(X_test, y_test))
    print("Best KNN Params:", knn_search.best_params_)

    results = {
        "best_accuracy": knn_search.score(X_test, y_test),
        "best_params": knn_search.best_params_
    }

    return results 
   
def NB(X_train, X_test, y_train, y_test):
    # --- Randomized Search for MultinomialNB ---
    nb_pipeline = Pipeline([
        ('classifier', MultinomialNB())
    ])

    # Parameters for MultinomialNB
    nb_param_dist = {
        'classifier__alpha': uniform(0.0, 1.0)
    }

    nb_search = RandomizedSearchCV(
        nb_pipeline,
        param_distributions=nb_param_dist,
        n_iter=10,
        cv=5,
        n_jobs=-1,
        random_state=42
    )

    print("\nFitting NB with RandomizedSearchCV...")
    with tqdm_joblib(tqdm.tqdm(total=n_iterations * cv_folds, desc="NB Search Progress")):
        nb_search.fit(X_train, y_train)

    print("Best NB Accuracy:", nb_search.score(X_test, y_test))
    print("Best NB Params:", nb_search.best_params_)

    results = {
        "best_accuracy": nb_search.score(X_test, y_test),
        "best_params": nb_search.best_params_
    }

    return results

def LR(X_train, X_test, y_train, y_test):
    lr_pipeline = Pipeline([
        ('classifier', LogisticRegression(solver='saga', max_iter=max_iter))
    ])

    lr_param_dist = {
        'classifier__C': loguniform(1e-4, 1e2),
        'classifier__tol': loguniform(1e-4, 1e-1)
    }

    lr_search = RandomizedSearchCV(
        lr_pipeline,
        param_distributions=lr_param_dist,
        n_iter=n_iterations,
        cv=cv_folds,
        n_jobs=-1
    )

    print("\nFitting Logistic Regression with RandomizedSearchCV...")
    with tqdm_joblib(tqdm.tqdm(total=n_iterations * cv_folds, desc="LR Search Progress")):
        lr_search.fit(X_train, y_train)

    print("Best LR Accuracy:", lr_search.score(X_test, y_test))
    print("Best LR Params:", lr_search.best_params_)

    results = {
        "best_accuracy": lr_search.score(X_test, y_test),
        "best_params": lr_search.best_params_
    }

    return results 

def main():
    #MNIST_search()
    #EMNIST_search()
    review_search()

if __name__ == "__main__":
    main()