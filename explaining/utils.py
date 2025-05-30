import json
from typing import Union, Callable

import torch


def load_vocab(path: str) -> dict:
    """
    Loads a vocabulary dictionary from a JSON file.

    Args:
        path (str): Path to the JSON file containing token-to-index mappings.

    Returns:
        dict: Vocabulary mapping from token (str) to index (int).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokens_to_words(token_ids, index_to_word, exclude_tokens={"<PAD>", "<UNK>"}):
    """
    Converts a list of token indices into readable words using an index-to-word mapping.

    Args:
        token_ids (Iterable[int]): List or tensor of token indices.
        index_to_word (dict): Mapping from index to word (int -> str).
        exclude_tokens (set, optional): Tokens to exclude from the output. Defaults to {"<PAD>", "<UNK>"}.

    Returns:
        list[str]: List of words corresponding to the given token IDs, excluding specified tokens.
    """
    words = [index_to_word.get(int(token), "[UNK]") for token in token_ids]
    return [w for w in words if w not in exclude_tokens]


def preprocess_text_to_tensor(
        text: str,
        vocab: dict,
        max_len: int,
        device: Union[str, torch.device],
        sequence_function: Callable,
) -> torch.LongTensor:
    """
    Converts input text into a tensor of token indices using the provided vocabulary and preprocessing function.

    Args:
        text (str): Raw input text to be tokenized and encoded.
        vocab (dict): Vocabulary mapping token (str) to index (int).
        max_len (int): Maximum sequence length (text will be padded or truncated).
        device (Union[str, torch.device]): Device to place the resulting tensor on.
        sequence_function (Callable): Function that converts text into a sequence of indices.

    Returns:
        torch.LongTensor: Tensor of shape (1, max_len) containing token indices.
    """
    sequence = sequence_function(text, vocab, max_len)
    tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    return tensor
