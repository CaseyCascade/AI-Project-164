# TODO This is chat GPT code just to demo gathering metrics for implementing part 4 
# It's basically useless but I'll gut it for structure lol 
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LogNorm
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

out_dir = "mnist_model_analysis_data"
results_dir = "Model Evaluation Data"
""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

def evaluate_model(name, model, X_test, y_test):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    # Classification Report (normalized metrics only)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Confusion Matrix (normalized only)
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

    # ROC-AUC & Curve
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    try:
        roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
        print(f"{name} ROC-AUC (OvR, macro): {roc_auc:.4f}")

        n_classes = y_score.shape[1]
        fpr, tpr = {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        for i in range(n_classes):
            ax2.plot(fpr[i], tpr[i], label=f"Class {i}")

        ax2.plot([0, 1], [0, 1], "k--", linewidth=1)

        ax2.set_xlim([0.8, 1.01])
        ax2.set_ylim([0.0, 0.2])
        ax2.set_title(f"{name} ROC Curves (Zoomed In)")
        ax2.set_xlabel("True Positive Rate (Recall)")
        ax2.set_ylabel("False Positive Rate")
        ax2.legend(loc="lower right", fontsize="small")
        ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        roc_img_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_roc.png")
        fig2.savefig(roc_img_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"→ Saved ROC curve plot to {roc_img_path}")

        # ROC data (normalized FPR/TPR only)
        roc_data = {
            "class_labels": list(range(n_classes)),
            "roc_auc_macro_ovr": roc_auc,
            "roc_curves": {
                str(i): {
                    "fpr": fpr[i].tolist(),
                    "tpr": tpr[i].tolist()
                }
                for i in range(n_classes)
            }
        }
        roc_json_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_roc_data.json")
        with open(roc_json_path, "w") as f:
            json.dump(roc_data, f, indent=2)
        print(f"→ Saved ROC curve data to {roc_json_path}")

    except ValueError:
        print(f"{name} ROC-AUC not available.")

    # Save classification report (already normalized by class)
    report_json_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_classification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"→ Saved classification report to {report_json_path}")

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
