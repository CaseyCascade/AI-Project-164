from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

from scipy.stats import loguniform, randint
from contextlib import contextmanager
import joblib
import warnings
import tqdm
import json
from pathlib import Path


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

# Load and preprocess MNIST
n_iterations = 10 
cv_folds = 3 

mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Standardize
scaler = StandardScaler()

# --- Randomized Search for Logistic Regression ---
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='saga', max_iter=1000))
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


# --- Randomized Search for KNN ---
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

knn_param_dist = {
    'classifier__n_neighbors': randint(3, 15),
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

# Prepare results for saving
results = {
    "logistic_regression": {
        "best_accuracy": lr_search.score(X_test, y_test),
        "best_params": lr_search.best_params_
    },
    "knn": {
        "best_accuracy": knn_search.score(X_test, y_test),
        "best_params": knn_search.best_params_
    }
}

# Save to JSON file
output_path = Path("search_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults written to {output_path.resolve()}")

