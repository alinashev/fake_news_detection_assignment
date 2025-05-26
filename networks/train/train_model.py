import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from metrics.model_metrics import ModelEvaluation
from metrics.display import (
    display_confusion_matrix,
    display_metrics,
    display_roc_auc,
    display_training_curves
)


def choose_device(preferred_device=None):
    """
        Chooses the computation device for model training or inference.

        Args:
            preferred_device (str, optional): Preferred device to use ("cuda", "cpu", etc.).
                                              If None, automatically selects CUDA if available.

        Returns:
            torch.device: The selected device.
    """

    if preferred_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(preferred_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            device = torch.device("cpu")
    return device


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
        Trains the model for a single epoch.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for the training set.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to run training on (CPU or GPU).

        Returns:
            tuple:
                - float: Average training loss for the epoch.
                - list: Ground truth labels.
                - list: Predicted binary labels (0 or 1).
                - list: Raw predicted probabilities.
    """

    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    progress_bar = tqdm(train_loader, desc="Training", leave=True, dynamic_ncols=True,
                        bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]")

    for batch in progress_bar:
        title_batch = batch["title"].to(device)
        text_batch = batch["text"].to(device)
        labels = batch["label"].to(device).float()

        optimizer.zero_grad()
        outputs = model(title_batch, text_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(outputs.detach().cpu().numpy())

        avg_loss = total_loss / (len(all_probs) or 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    predictions = (np.array(all_probs) > 0.5).astype(int)
    avg_loss = total_loss / len(train_loader)
    return avg_loss, all_labels, predictions, all_probs


def evaluate_model(model, data_loader, criterion, device):
    """
        Evaluates the model on a given dataset (typically validation or test set).

        Args:
            model (torch.nn.Module): The trained PyTorch model.
            data_loader (DataLoader): DataLoader for the explaining set.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to run explaining on.

        Returns:
            tuple:
                - float: Average loss over the explaining set.
                - list: Ground truth labels.
                - list: Predicted binary labels (0 or 1).
                - list: Raw predicted probabilities.
    """

    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in data_loader:
            title_batch = batch["title"].to(device)
            text_batch = batch["text"].to(device)
            labels = batch["label"].to(device)

            outputs = model(title_batch, text_batch)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = outputs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            all_labels.extend(labels_np)
            all_probs.extend(probs)

    predictions = (np.array(all_probs) > 0.5).astype(int)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_labels, predictions, all_probs


def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-3, device=None, patience=3):
    """
        Trains the model for multiple epochs with early stopping and displays performance metrics.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            num_epochs (int, optional): Number of epochs to train. Defaults to 5.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            device (str or torch.device, optional): Device to train the model on. Defaults to None (auto-select).
            patience (int, optional): Number of epochs to wait before early stopping if no improvement. Defaults to 3.

        Returns:
            tuple:
                - model (torch.nn.Module): The trained model with the best weights loaded.
                - dict: Training history containing 'train_loss', 'val_loss', and 'val_f1' per epoch.
    """

    device = choose_device(device)
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0
    best_model_state = None
    early_stopping_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": []
    }

    for epoch in range(num_epochs):
        train_loss, train_labels, train_preds, train_probs = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_labels, val_preds, val_probs = evaluate_model(
            model, val_loader, criterion, device
        )
        eval_metrics = ModelEvaluation(
            y_train=train_labels, train_prediction=train_preds, train_pr_proba=train_probs,
            y_val=val_labels, val_prediction=val_preds, val_pr_proba=val_probs
        )
        val_f1 = next((m.val_value for m in eval_metrics.compute_all_metrics() if m.name == "f1_score"), 0)

        print(f"\n|Epoch {epoch + 1}/{num_epochs}| Train Loss: {train_loss:.4f}; "
              f"Val Loss: {val_loss:.4f}; Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping patience: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

    print(f"\nBest Val F1: {best_val_f1:.4f}")
    if best_model_state:
        model.load_state_dict(best_model_state)

    display_metrics(eval_metrics)
    display_confusion_matrix(
        y_train=train_labels, y_val=val_labels,
        train_pred=train_preds, val_pred=val_preds
    )
    display_roc_auc(
        y_train=train_labels, y_val=val_labels,
        train_pr_proba=train_probs, val_pr_proba=val_probs
    )
    display_training_curves(history)

    return model, history
