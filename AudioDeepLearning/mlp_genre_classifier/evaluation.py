import matplotlib.pyplot as plt
import torch


def evaluate(model, loader, criterion, device):
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	with torch.no_grad():
		for inputs, targets in loader:
			inputs = inputs.to(device)
			targets = targets.to(device)
			logits = model(inputs)
			total_loss += criterion(logits, targets).item() * inputs.size(0)
			total_correct += (logits.argmax(dim=1) == targets).sum().item()
			total_samples += inputs.size(0)

	return {
		"loss": total_loss / total_samples,
		"accuracy": total_correct / total_samples,
	}


def plot_training_history(history, output_path: str = "training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))

    loss_axis.plot(epochs, history["train_loss"], label="Entraînement")
    loss_axis.plot(epochs, history["validation_loss"], label="Validation")
    loss_axis.set_title("Évolution de la perte")
    loss_axis.set_xlabel("Époque")
    loss_axis.set_ylabel("Cross-entropy")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)

    accuracy_axis.plot(epochs, history["train_accuracy"], label="Entraînement")
    accuracy_axis.plot(epochs, history["validation_accuracy"], label="Validation")
    accuracy_axis.set_title("Évolution de la précision")
    accuracy_axis.set_xlabel("Époque")
    accuracy_axis.set_ylabel("Précision")
    accuracy_axis.legend()
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)

    plt.close(figure)
