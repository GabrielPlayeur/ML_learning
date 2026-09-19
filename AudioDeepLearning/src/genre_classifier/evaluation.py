import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def collect_predictions(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * inputs.size(0)
            total_samples += inputs.size(0)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    return {
        "loss": total_loss / total_samples,
        "targets": np.concatenate(all_targets),
        "predictions": np.concatenate(all_predictions),
        "probabilities": np.concatenate(all_probabilities),
    }


def evaluate(model, loader, criterion, device):
    outputs = collect_predictions(model, loader, criterion, device)
    targets = outputs["targets"]
    predictions = outputs["predictions"]
    probabilities = outputs["probabilities"]

    try:
        roc_auc = roc_auc_score(
            targets,
            probabilities,
            multi_class="ovr",
            labels=np.arange(probabilities.shape[1]),
        )
    except ValueError:
        roc_auc = float("nan")

    return {
        "loss": outputs["loss"],
        "accuracy": float(np.mean(predictions == targets)),
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(targets, predictions, average="weighted", zero_division=0),
        "roc_auc": roc_auc,
        "targets": targets,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def plot_training_history(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, (loss_axis, accuracy_axis, f1_axis) = plt.subplots(1, 3, figsize=(18, 4))

    loss_axis.plot(epochs, history["train_loss"], label="Entraînement")
    loss_axis.plot(epochs, history["validation_loss"], label="Validation")
    loss_axis.set_title("Évolution de la perte")
    loss_axis.set_xlabel("Époque")
    loss_axis.set_ylabel("Cross-entropy")
    loss_axis.legend()
    loss_axis.set_ylim(bottom=0)
    loss_axis.grid(alpha=0.3)

    accuracy_axis.plot(epochs, history["train_accuracy"], label="Entraînement")
    accuracy_axis.plot(epochs, history["validation_accuracy"], label="Validation")
    accuracy_axis.set_title("Évolution de la précision")
    accuracy_axis.set_xlabel("Époque")
    accuracy_axis.set_ylabel("Précision")
    accuracy_axis.legend()
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.grid(alpha=0.3)

    f1_axis.plot(epochs, history["train_f1_macro"], label="Entraînement")
    f1_axis.plot(epochs, history["validation_f1_macro"], label="Validation")
    f1_axis.set_title("Évolution du F1 macro")
    f1_axis.set_xlabel("Époque")
    f1_axis.set_ylabel("F1 macro")
    f1_axis.legend()
    f1_axis.set_ylim(0, 1)
    f1_axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)

    plt.close(figure)


def save_evaluation_report(metrics, class_names, output_path):
    report = classification_report(
        metrics["targets"],
        metrics["predictions"],
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    serializable_metrics = {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "roc_auc": metrics["roc_auc"],
        "classification_report": report,
    }

    import json
    with open(output_path, "w") as file:
        json.dump(serializable_metrics, file, indent=2, allow_nan=True)


def plot_confusion_matrix(metrics, class_names, output_path):
    matrix = confusion_matrix(
        metrics["targets"],
        metrics["predictions"],
        labels=np.arange(len(class_names)),
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xlabel="Classe prédite",
        ylabel="Classe réelle",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Matrice de confusion",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_roc_curves(metrics, class_names, output_path):
    figure, axis = plt.subplots(figsize=(8, 7))
    targets = metrics["targets"]
    probabilities = metrics["probabilities"]
    for class_index, class_name in enumerate(class_names):
        binary_targets = (targets == class_index).astype(int)
        if binary_targets.min() == binary_targets.max():
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_targets,
            probabilities[:, class_index],
        )
        axis.plot(false_positive_rate, true_positive_rate, label=class_name)
    axis.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    axis.set(
        xlabel="Taux de faux positifs",
        ylabel="Taux de vrais positifs",
        title="Courbes ROC par classe",
    )
    axis.legend(loc="lower right", fontsize="small")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
