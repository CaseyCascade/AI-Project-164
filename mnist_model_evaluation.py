# TODO This is chat GPT code just to demo gathering metrics for implementing part 4 
# It's basically useless but I'll gut it for structure lol 
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

out_dir = "mnist_graphs"

def evaluate_model(name, model, X_test, y_test):
    # ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)

    # 1) Predict
    y_pred = model.predict(X_test)

    # 2) Classification report
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred))

    # 3) Confusion matrix (normalized rows)
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    # 4) Plot + save confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_title(f"{name} Confusion Matrix", pad=20)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    out_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_confusion.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Saved confusion matrix to {out_path}")

    # 5) ROC‐AUC & curves
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    try:
        roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
        print(f"{name} ROC-AUC (OvR, macro): {roc_auc:.4f}")

        # compute curves for each class
        n_classes = y_score.shape[1]
        fpr, tpr = {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])

        # Plot all curves on one figure
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        for i in range(n_classes):
            ax2.plot(tpr[i], fpr[i], label=f"Class {i}")
        ax2.plot([0, 1], [0, 1], "k--", linewidth=1)

        ax2.set_title(f"{name} ROC Curves (TPR vs FPR)")
        ax2.set_xlabel("True Positive Rate (Recall)")
        ax2.set_ylabel("False Positive Rate")
        ax2.legend(loc="lower right", fontsize="small")
        ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        roc_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_roc.png")
        fig2.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"→ Saved ROC curves to {roc_path}")

    except ValueError:
        print(f"{name} ROC-AUC not available.")

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
