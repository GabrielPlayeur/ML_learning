import json
import copy
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

if __name__ == "__main__":
    print("Loading the dataset")
    inputs, targets = load_data("data/data.json")
    # (batch_size, 130, 13)

    print("Spliting and std the training data")
    inputs_train, inputs_temp, targets_train, targets_temp = train_test_split(
        inputs,
        targets,
        test_size=0.3,
        random_state=42,
        stratify=targets,
    )
    inputs_validation, inputs_test, targets_validation, targets_test = train_test_split(
        inputs_temp,
        targets_temp,
        test_size=0.5,
        random_state=42,
        stratify=targets_temp,
    )

    # STD
    mean = inputs_train.mean(axis=(0, 1), keepdims=True)
    std = inputs_train.std(axis=(0, 1), keepdims=True)
    inputs_train = (inputs_train - mean) / (std + 1e-8)
    inputs_validation = (inputs_validation - mean) / (std + 1e-8)
    inputs_test = (inputs_test - mean) / (std + 1e-8)

    train_inputs = torch.tensor(inputs_train, dtype=torch.float32)
    train_targets = torch.tensor(targets_train, dtype=torch.long)
    validation_inputs = torch.tensor(inputs_validation, dtype=torch.float32)
    validation_targets = torch.tensor(targets_validation, dtype=torch.long)
    test_inputs = torch.tensor(inputs_test, dtype=torch.float32)
    test_targets = torch.tensor(targets_test, dtype=torch.long)

    model = nn.Sequential(
        nn.Flatten(),

        nn.Linear(130 * 13, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),

        # nn.Linear(512, 256),
        # nn.ReLU(),
        # nn.Dropout(p=0.3),

        nn.Linear(128, 64),
        nn.ReLU(),

        nn.Linear(64, 10)
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=32, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("Start Learning")
    epochs = 50
    best_validation_accuracy = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_inputs.size(0)

        model.eval()
        with torch.no_grad():
            train_logits = model(train_inputs.to(device))
            train_predictions = train_logits.argmax(dim=1)
            train_accuracy = (train_predictions == train_targets.to(device)).float().mean()

            validation_logits = model(validation_inputs.to(device))
            validation_predictions = validation_logits.argmax(dim=1)
            validation_accuracy = (
                validation_predictions == validation_targets.to(device)
            ).float().mean()

        if validation_accuracy.item() > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy.item()
            best_model_state = copy.deepcopy(model.state_dict())

        average_loss = total_loss / len(train_inputs)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {average_loss:.4f} - "
            f"train accuracy: {train_accuracy.item():.4f} - "
            f"validation accuracy: {validation_accuracy.item():.4f}"
        )

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(test_inputs.to(device))
        test_predictions = test_logits.argmax(dim=1)
        test_accuracy = (test_predictions == test_targets.to(device)).float().mean()

    print(
        f"Best validation accuracy: {best_validation_accuracy:.4f} - "
        f"test accuracy: {test_accuracy.item():.4f}"
    )
