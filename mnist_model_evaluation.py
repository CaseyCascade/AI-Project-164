# TODO This is chat GPT code just to demo gathering metrics for implementing part 4 
# It's basically useless but I'll gut it for structure lol 

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def evaluate_model(name, model, X_test, y_test):
    """
    Prints classification report, confusion matrix, and ROC-AUC
    for a trained multiclass model.
    """
    # 1) Predict labels
    y_pred = model.predict(X_test)

    # 2) Classification report: precision, recall, F1-score
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred))

    # 3) Confusion matrix (as array)
    cm = confusion_matrix(y_test, y_pred)
    print(f"--- {name} Confusion Matrix ---\n{cm}")

    # 4) Plot confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_title(f"{name} Confusion Matrix", pad=20)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.show()

    # 5) ROC-AUC (One-vs-Rest)
    #    - Use predict_proba if available, else decision_function
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    try:
        roc_auc = roc_auc_score(
            y_test, y_score, multi_class="ovr", average="macro"
        )
        print(f"{name} ROC-AUC (OvR, macro-avg): {roc_auc:.4f}")

        # Optional: plot ROC curves for each class
        # Compute fpr/tpr for each label
        fpr = dict()
        tpr = dict()
        n_classes = y_score.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(
                (y_test == i).astype(int), y_score[:, i]
            )

        plt.figure(figsize=(6, 5))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"Class {i}")
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.title(f"{name} ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right", fontsize='small')
        plt.show()

    except ValueError:
        print(f"{name} ROC-AUC not available (check predict_proba/decision_function).")

def LR_test(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(
        solver='saga',
        max_iter=100,
        tol=0.00127722364893473,
        C=1.6176755124355395,
    )
    lr.fit(X_train, y_train)
    evaluate_model("Logistic Regression", lr, X_test, y_test)

def KNN_test(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    evaluate_model("K-Nearest Neighbors", knn, X_test, y_test)

def main():
    # 1) Load MNIST
    mnist = fetch_openml("mnist_784", as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # 3) Run evaluations
    LR_test(X_train, X_test, y_train, y_test)
    KNN_test(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
