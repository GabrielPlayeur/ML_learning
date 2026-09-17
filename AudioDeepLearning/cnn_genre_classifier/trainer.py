
import copy

from .evaluation import evaluate

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        return total_loss / total_samples

    def fit(self, train_loader, validation_loader, epochs, patience, min_delta):
        history = {
            "train_loss": [],
            "validation_loss": [],
            "train_accuracy": [],
            "validation_accuracy": [],
        }
        best_accuracy = float("-inf")
        best_model_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            train_metrics = evaluate(
                self.model, train_loader, self.criterion, self.device
            )
            validation_metrics = evaluate(
                self.model, validation_loader, self.criterion, self.device
            )

            self.scheduler.step(validation_metrics["loss"])

            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["validation_accuracy"].append(validation_metrics["accuracy"])

            validation_accuracy = validation_metrics["accuracy"]
            if validation_accuracy > best_accuracy + min_delta:
                best_accuracy = validation_accuracy
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"loss: {train_loss:.4f} - "
                f"train accuracy: {train_metrics['accuracy']:.4f} - "
                f"validation accuracy: {validation_accuracy:.4f}"
            )

            if epochs_without_improvement >= patience:
                print(f"Early stopping: aucune amélioration depuis {patience} époques.")
                break

        self.model.load_state_dict(best_model_state)
        return history, best_accuracy

    def validate(self, loader):
        return evaluate(self.model, loader, self.criterion, self.device)