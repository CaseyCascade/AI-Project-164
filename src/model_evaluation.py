import os
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LogNorm
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = "Model Analysis"
MNIST_DIR = "MNIST"
REVIEW_DIR = "Review"
EMNIST_DIR = "EMNIST" 
KNN_DIR = "K-Nearest Neighbors"
NB_DIR = "Naive Bayes"
LR_DIR = "Logistic Regression"

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def classification_report_evaluation(name, model, X_test, y_test, out_dir):
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Classification report not available. Reason: {e}")
        return None

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_json_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_classification_report.json")
    
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"→ Saved classification report to {report_json_path}")
    return y_pred

def roc_evaluation(name, model, X_test, y_test, class_labels, out_dir):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    try:
        if y_score.ndim > 1 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
        else:
            roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")

        print(f"{name} ROC-AUC: {roc_auc:.4f}")

        if y_score.ndim == 1:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.plot(fpr, tpr, label="ROC Curve")
            ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title(f"{name} ROC Curve")
            ax2.legend(loc="lower right")
            roc_img_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_roc.png")
            fig2.savefig(roc_img_path, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            print(f"→ Saved ROC curve plot to {roc_img_path}")
        
        else:
            n_classes = y_score.shape[1]
            fpr, tpr = {}, {}
            for i in range(n_classes):
                try:
                    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
                except ValueError:
                    print(f"⚠️  Skipping ROC for class {i}")
                    continue

            fig2, ax2 = plt.subplots(figsize=(6, 6))
            for i in fpr:
                label_name = class_labels[i] if class_labels else str(i)
                ax2.plot(fpr[i], tpr[i], label=f"Class {label_name}")

            ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax2.set_xlim([0.0, 0.5])
            ax2.set_ylim([0.5, 1.0])
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate (Recall)")
            ax2.set_title(f"{name} ROC Curves (Zoomed In)")
            if len(fpr) <= 10:
                ax2.legend(loc="lower right", fontsize="small")
            roc_img_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_roc.png")
            fig2.savefig(roc_img_path, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            print(f"→ Saved ROC curves plot to {roc_img_path}")

    except ValueError as e:
        print(f"{name} ROC-AUC not available. Reason: {e}")

def confusion_matrix_evaluation(name, y_pred, y_test, class_labels, out_dir):
    minimal_display = False
    if len(class_labels) > 15: 
        minimal_display = True 
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    cm_data = {
        "normalized": cm_norm.tolist(),
        "class_labels": class_labels
    }
    cm_json_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_confusion_data.json")
    with open(cm_json_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"→ Saved normalized confusion matrix to {cm_json_path}")

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_title(f"{name} Confusion Matrix", pad=20)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)

    if not minimal_display:
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                val = cm_norm[i, j]
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

    cm_img_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_confusion.png")
    fig.savefig(cm_img_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Saved confusion matrix plot to {cm_img_path}")

def LR_test(X_train, X_test, y_train, y_test, class_labels, out_dir, solver, max_iter, tol, C):
    name = "Logistic Regression"
    out_dir = os.path.join(out_dir, LR_DIR)
    os.makedirs(out_dir, exist_ok=True)

    lr = LogisticRegression(
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        C=C,
    )
    lr.fit(X_train, y_train)

    y_pred = classification_report_evaluation(name, lr, X_test, y_test, out_dir)
    roc_evaluation(name, lr, X_test, y_test, class_labels, out_dir)
    confusion_matrix_evaluation(name, y_pred, y_test, class_labels, out_dir)

def KNN_test(X_train, X_test, y_train, y_test, class_labels, out_dir, n_neighbors, p=2, weights="uniform"):
    name = "KNN"
    out_dir = os.path.join(out_dir, KNN_DIR)
    os.makedirs(out_dir, exist_ok=True)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, weights=weights)
    knn.fit(X_train, y_train)

    y_pred = classification_report_evaluation(name, knn, X_test, y_test, out_dir)
    roc_evaluation(name, knn, X_test, y_test, class_labels, out_dir)
    confusion_matrix_evaluation(name, y_pred, y_test, class_labels, out_dir)

def NB_test(X_train, X_test, y_train, y_test, class_labels, out_dir):
    name = "Naive Bayes"
    out_dir = os.path.join(out_dir, NB_DIR)
    os.makedirs(out_dir, exist_ok=True)

    nb = GaussianNB()
    nb.fit(X_train.toarray() if hasattr(X_train, "toarray") else X_train, y_train)

    y_pred = classification_report_evaluation(name, nb, X_test.toarray() if hasattr(X_test, "toarray") else X_test, y_test, out_dir)
    roc_evaluation(name, nb, X_test.toarray() if hasattr(X_test, "toarray") else X_test, y_test, class_labels, out_dir)
    confusion_matrix_evaluation(name, y_pred, y_test, class_labels, out_dir)

def MNIST_evaluation():
    mnist = fetch_openml("mnist_784", as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    out_dir = os.path.join(RESULTS_DIR, MNIST_DIR)
    os.makedirs(out_dir, exist_ok=True)
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    LR_test(X_train, X_test, y_train, y_test, class_labels, out_dir, 'saga', 100, 0.00127722364893473, 1.6176755124355395)
    NB_test(X_train, X_test, y_train, y_test, class_labels, out_dir)
    KNN_test(X_train, X_test, y_train, y_test, class_labels, out_dir, 5)

def review_evaluation():
    out_dir = os.path.join(RESULTS_DIR, REVIEW_DIR)
    os.makedirs(out_dir, exist_ok=True)
    dataset = load_dataset("Kwaai/IMDB_Sentiment", split="train").shuffle(seed=42)
    dataset = dataset.select(range(1000))
    texts = dataset["text"]
    labels = dataset["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = np.ravel(np.array(labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_labels = ["Negative", "Positive"]

    LR_test(X_train, X_test, y_train, y_test, class_labels, out_dir, 'saga', 100, 0.00127722364893473, 1.6176755124355395)
    NB_test(X_train, X_test, y_train, y_test, class_labels, out_dir)
    KNN_test(X_train, X_test, y_train, y_test, class_labels, out_dir, 5)

def EMNIST_evaluation():
    emnist = fetch_openml("EMNIST_Balanced", version=1, as_frame=False)
    X, y = emnist.data, emnist.target.astype(int)

    X = X / 255.0

    USE_PCA = True
    if USE_PCA:
        pca = PCA(n_components=100)
        X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    out_dir = os.path.join(RESULTS_DIR, EMNIST_DIR)
    os.makedirs(out_dir, exist_ok=True)

    class_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
    ]

    LR_test(X_train, X_test, y_train, y_test, class_labels, out_dir, 'saga', 100, 0.00127722364893473, 1.6176755124355395)
    NB_test(X_train, X_test, y_train, y_test, class_labels, out_dir)
    KNN_test(X_train, X_test, y_train, y_test, class_labels, out_dir, 5)

def main():
    MNIST_evaluation()
    EMNIST_evaluation()
    review_evaluation()

if __name__ == "__main__":
    main()
