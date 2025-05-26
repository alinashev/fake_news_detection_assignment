import torch
import json
from networks.utils.loader import load_model_state

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding(path):
    """
    Loads a pre-trained embedding matrix from a file.

    Args:
        path (str): Path to the `.pt` or `.pth` file containing the embedding matrix.

    Returns:
        torch.Tensor: Loaded embedding tensor of shape (vocab_size, embedding_dim).
    """
    return torch.load(path)


def load_model(model_class, embedding_matrix, model_path):
    """
    Loads the trained model with its weights and initialized embedding layer.

    Args:
        model_class (Type[torch.nn.Module]): The class of the model to instantiate.
        embedding_matrix (torch.Tensor): Pre-trained embedding matrix of shape (vocab_size, embedding_dim).
        model_path (str): Path to the saved model checkpoint file (.pt or .pth).

    Returns:
        torch.nn.Module: Instantiated and loaded model ready for inference on the specified device.
    """
    model = load_model_state(
        model_class=model_class,
        embedding_tensor=embedding_matrix,
        path=model_path,
        device=DEVICE
    )
    return model


def load_data(data_path):
    """
    Loads the preprocessed dataset from a file.

    Args:
        data_path (str): Path to the `.pt` or `.pth` file with preprocessed data.

    Returns:
        dict: Dictionary containing input tensors (e.g., "X_title", "X_text").
    """
    return torch.load(data_path)


def load_vocab(path):
    """
    Loads the vocabulary mapping from a JSON file.

    Args:
        path (str): Path to the JSON file containing the vocabulary.

    Returns:
        dict: Mapping from token to index (token -> int).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
