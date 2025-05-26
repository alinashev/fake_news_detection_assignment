import torch
import torch.nn as nn

class FakeNewsLstmClassifier(nn.Module):
    """
    BiLSTM-based classifier for fake news detection using separate encoders
    for title and text inputs.

    This model uses pre-trained embeddings and bidirectional LSTMs to encode
    title and body text separately, then concatenates the representations and
    classifies the result using a linear layer.

    Args:
        embedding_matrix (torch.Tensor): Pre-trained embedding matrix of shape (vocab_size, embedding_dim).
        hidden_size (int, optional): Number of hidden units in each LSTM direction. Defaults to 128.
        dropout (float, optional): Dropout probability applied before the output layer. Defaults to 0.3.
    """

    def __init__(self, embedding_matrix, hidden_size=128, dropout=0.3):
        super(FakeNewsLstmClassifier, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, padding_idx=0, freeze=False
        )

        self.title_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.text_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 4, 1)

    def _encode_title(self, x):
        """
        Encodes the title input using a bidirectional LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) for title tokens.

        Returns:
            torch.Tensor: Concatenated forward and backward hidden states (batch_size, 2 * hidden_size).
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.title_encoder(embedded)
        return torch.cat((hidden[0], hidden[1]), dim=1)

    def _encode_text(self, x):
        """
        Encodes the text input using a bidirectional LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) for text tokens.

        Returns:
            torch.Tensor: Concatenated forward and backward hidden states (batch_size, 2 * hidden_size).
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.text_encoder(embedded)
        return torch.cat((hidden[0], hidden[1]), dim=1)

    def forward(self, title_input, text_input):
        """
        Performs a forward pass of the model.

        Args:
            title_input (torch.Tensor): Title token sequences (batch_size, title_seq_len).
            text_input (torch.Tensor): Text token sequences (batch_size, text_seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size,) for binary classification.
        """
        title_repr = self._encode_title(title_input)
        text_repr = self._encode_text(text_input)
        combined = torch.cat((title_repr, text_repr), dim=1)
        x = self.dropout(combined)
        return self.fc(x).squeeze(1)
