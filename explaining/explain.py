from captum.attr import configure_interpretable_embedding_layer

from explaining.loader import load_embedding, load_model, load_data, load_vocab
from explaining.wrappers import WrapperTitleBranchModel, WrapperTextBranchModel


def load_all_artifacts(model_class, embedding_path, model_path, data_path, vocab_path):
    """
    Loads all necessary artifacts: embedding matrix, model, data, and vocabulary.

    Args:
        model_class (Type[torch.nn.Module]): Class of the model to be instantiated.
        embedding_path (str): Path to the embedding file (.pt or .pth).
        model_path (str): Path to the saved model weights file.
        data_path (str): Path to the preprocessed dataset file (.pt).
        vocab_path (str): Path to the vocabulary JSON file.

    Returns:
        Tuple[torch.nn.Module, dict, dict]:
            - Loaded and initialized model.
            - Dataset dictionary with input tensors.
            - Mapping from index to word for interpretation.
    """
    embedding_matrix = load_embedding(embedding_path)
    model = load_model(model_class, embedding_matrix, model_path)
    data = load_data(data_path)
    vocab = load_vocab(vocab_path)
    index_to_word = {v: k for k, v in vocab.items()}
    return model, data, index_to_word



def get_validation_inputs(data, idx, device):
    """
    Retrieves validation input tensors for a specific example index.

    Args:
        data (dict): Dictionary containing 'X_title' and 'X_text' tensors.
        idx (int): Index of the validation example to retrieve.
        device (torch.device): The device to move tensors to ("cpu" or "cuda").

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Title and text input tensors for the specified index, and full validation sets.
    """
    X_title = data["X_title"]
    X_text = data["X_text"]
    train_size = int(len(X_title) * 0.7)
    X_val_title = X_title[train_size:]
    X_val_text = X_text[train_size:]
    title_input_ids = X_val_title[idx].unsqueeze(0).to(device)
    text_input_ids = X_val_text[idx].unsqueeze(0).to(device)
    return title_input_ids, text_input_ids, X_val_title, X_val_text


def prepare_embeddings(model, title_input_ids, text_input_ids):
    """
    Configures interpretable embeddings and computes embedded inputs.

    Args:
        model (torch.nn.Module): The trained model with an `embedding` layer.
        title_input_ids (torch.Tensor): Token indices for the title input.
        text_input_ids (torch.Tensor): Token indices for the text input.

    Returns:
        Tuple[InterpretableEmbeddingBase, torch.Tensor, torch.Tensor]:
            Captum interpretable embedding layer and embedded tensors for title and text.
    """
    interpretable_emb = configure_interpretable_embedding_layer(model, "embedding")
    embedded_title = interpretable_emb.indices_to_embeddings(title_input_ids)
    embedded_text = interpretable_emb.indices_to_embeddings(text_input_ids)
    return interpretable_emb, embedded_title, embedded_text


def get_wrapper_and_tokens(model, embedded_title, embedded_text, X_val_title, X_val_text, idx, mode):
    """
    Selects the appropriate wrapper model and token sequence for interpretation.

    Args:
        model (torch.nn.Module): The trained model.
        embedded_title (torch.Tensor): Embedded input for title.
        embedded_text (torch.Tensor): Embedded input for text.
        X_val_title (torch.Tensor): Full validation title inputs.
        X_val_text (torch.Tensor): Full validation text inputs.
        idx (int): Index of the sample to explain.
        mode (str): Which branch to explain ("title" or "text").

    Returns:
        Tuple[torch.nn.Module, torch.Tensor, torch.Tensor, str]:
            Wrapped model with one frozen branch, input token sequence, embedded input, and label ("Title" or "Text").
    """
    if mode == "title":
        return WrapperTitleBranchModel(model, embedded_text), X_val_title[idx], embedded_title, "Title"
    else:
        return WrapperTextBranchModel(model, embedded_title), X_val_text[idx], embedded_text, "Text"
