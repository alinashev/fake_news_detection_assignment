import torch
import torch.nn as nn

class FakeNewsRNNClassifier(nn.Module):
    """
    GRU-based classifier for fake news detection using separate encoders
    for title and text inputs.

    This model embeds input tokens using pre-trained embeddings, encodes the title and text
    using separate GRUs, and combines their representations for binary classification.

    Args:
        embedding_matrix (torch.Tensor): Pre-trained embedding matrix of shape (vocab_size, embedding_dim).
        hidden_size (int, optional): Number of hidden units in each GRU. Defaults to 128.
        dropout (float, optional): Dropout probability applied before the final linear layer. Defaults to 0.3.
    """

    def __init__(self, embedding_matrix, hidden_size=128, dropout=0.3):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)

        self.title_encoder = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.text_encoder = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def _encode_title(self, x):
        """
        Encodes the title input using a unidirectional GRU.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, title_seq_len).

        Returns:
            torch.Tensor: Last hidden state of shape (batch_size, hidden_size).
        """
        embedded = self.embedding(x)
        _, hidden = self.title_encoder(embedded)
        return hidden.squeeze(0)

    def _encode_text(self, x):
        """
        Encodes the text input using a unidirectional GRU.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, text_seq_len).

        Returns:
            torch.Tensor: Last hidden state of shape (batch_size, hidden_size).
        """
        embedded = self.embedding(x)
        _, hidden = self.text_encoder(embedded)
        return hidden.squeeze(0)

    def forward(self, title_input, text_input):
        """
        Performs a forward pass through the model.

        Args:
            title_input (torch.Tensor): Input tensor for titles (batch_size, title_seq_len).
            text_input (torch.Tensor): Input tensor for texts (batch_size, text_seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size,), to be passed through sigmoid for binary classification.
        """
        title_repr = self._encode_title(title_input)
        text_repr = self._encode_text(text_input)
        combined = torch.cat((title_repr, text_repr), dim=1)
        x = self.dropout(combined)
        return self.fc(x).squeeze(1)
