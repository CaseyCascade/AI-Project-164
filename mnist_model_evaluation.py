import os
import warnings
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LogNorm
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = "Model Analysis"
MNIST_DIR = "MNIST"
FOOD_DIR = "Food"
EMNIST_DIR = "EMNIST" #? Not sure if we're using this one yet #TODO
KNN_DIR = "KNN"
LR_DIR = "Logistic Regression"

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def classification_report_evaluation(name, model, X_test, y_test, out_dir):
        y_pred = model.predict(X_test)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_json_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_classification_report.json")
        with open(report_json_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"→ Saved classification report to {report_json_path}")
        return y_pred

def roc_evalutation(name, model, X_test, y_test, out_dir):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    try:
        roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
        print(f"{name} ROC-AUC (OvR, macro): {roc_auc:.4f}")

        n_classes = y_score.shape[1]
        fpr, tpr = {}, {}

        # Attempt to compute ROC curve for each class individually
        for i in range(n_classes):
            try:
                fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
            except ValueError:
                print(f"⚠️  Skipping ROC for class {i} (not enough positive/negative samples in y_test)")
                continue

        # Plot only the valid ROC curves
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        for i in fpr:  # only plot classes that succeeded
            ax2.plot(fpr[i], tpr[i], label=f"Class {i}")

        ax2.plot([0, 1], [0, 1], "k--", linewidth=1)

        # Adjust zoom (standard ROC layout)
        ax2.set_xlim([0.0, 0.5])
        ax2.set_ylim([0.5, 1.0])
        ax2.set_title(f"{name} ROC Curves (Zoomed In)")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate (Recall)")
        ax2.legend(loc="lower right", fontsize="small")
        ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        roc_img_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_roc.png")
        fig2.savefig(roc_img_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"→ Saved ROC curves plot to {roc_img_path}")

        # Save ROC data
        roc_data = {
            "class_labels": list(fpr.keys()),  # only classes that succeeded
            "roc_auc_macro_ovr": roc_auc,
            "roc_curves": {
                str(i): {
                    "fpr": fpr[i].tolist(),
                    "tpr": tpr[i].tolist()
                }
                for i in fpr
            }
        }
        roc_json_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_roc_data.json")
        with open(roc_json_path, "w") as f:
            json.dump(roc_data, f, indent=2)
        print(f"→ Saved ROC data to {roc_json_path}")

    except ValueError as e:
        print(f"{name} ROC-AUC not available. Reason: {e}")



def confusion_matrix_evaluation(name, y_pred, y_test, out_dir):
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    cm_data = {
        "normalized": cm_norm.tolist(),
        "class_labels": list(range(10))
    }
    cm_json_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_confusion_data.json")
    with open(cm_json_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"→ Saved normalized confusion matrix to {cm_json_path}")

    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm_norm + 1e-5, cmap="Blues", norm=LogNorm(vmin=1e-5, vmax=1))
    fig.colorbar(cax)
    ax.set_title(f"{name} Confusion Matrix", pad=20)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    for i in range(10):
        for j in range(10):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if val < 0.5 else "white")

    cm_img_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_confusion.png")
    fig.savefig(cm_img_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Saved confusion matrix plot to {cm_img_path}")


def LR_test(X_train, X_test, y_train, y_test, out_dir):
    name = "Logistic Regression"
    out_dir = out_dir + "/" + LR_DIR
    os.makedirs(out_dir, exist_ok=True)

    lr = LogisticRegression(
        solver='saga',
        max_iter=100,
        tol=0.00127722364893473,
        C=1.6176755124355395,
    )
    lr.fit(X_train, y_train)

    y_pred = classification_report_evaluation(name, lr, X_test, y_test, out_dir)
    roc_evalutation(name, lr, X_test, y_test, out_dir)
    confusion_matrix_evaluation(name, y_pred, y_test, out_dir)
    

def KNN_test(X_train, X_test, y_train, y_test, out_dir):
    name = "KNN"
    out_dir = out_dir + "/" + KNN_DIR
    os.makedirs(out_dir, exist_ok=True)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = classification_report_evaluation(name, knn, X_test, y_test, out_dir)
    roc_evalutation(name, knn, X_test, y_test, out_dir)
    confusion_matrix_evaluation(name, y_pred, y_test, out_dir)

def main():
    # 1) Load MNIST
    mnist = fetch_openml("mnist_784", as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # MNIST Evaluations
    out_dir = RESULTS_DIR + "/" + MNIST_DIR
    os.makedirs(out_dir, exist_ok=True)
    LR_test(X_train, X_test, y_train, y_test, out_dir)
    KNN_test(X_train, X_test, y_train, y_test, out_dir)

    # Food Evaluations #TODO
    out_dir = RESULTS_DIR + "/" + FOOD_DIR
    os.makedirs(out_dir, exist_ok=True)

    # EMNIST Evaluations #TODO
    out_dir = RESULTS_DIR + "/" + EMNIST_DIR
    os.makedirs(out_dir, exist_ok=True)

if __name__ == "__main__":
    main()
