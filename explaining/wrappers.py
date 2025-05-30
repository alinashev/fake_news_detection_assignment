import torch
import torch.nn as nn

class WrapperTitleBranchModel(nn.Module):
    """
    Wrapper model for analyzing the title branch of a dual-input model.

    This wrapper freezes the embedded text input and allows Captum or other
    attribution methods to analyze the title input exclusively.

    Args:
        model (nn.Module): Original dual-input model with separate encoders for title and text.
        embedded_text_input (torch.Tensor): Precomputed embedded tensor for the text input.
    """

    def __init__(self, model: nn.Module, embedded_text_input: torch.Tensor):
        super().__init__()
        self.model = model
        self.embedded_text_input = embedded_text_input

    def forward(self, embedded_title_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fixed text embedding and variable title input.

        Args:
            embedded_title_input (torch.Tensor): Embedded input tensor for the title branch, shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: Output predictions (probabilities), shape (batch_size, 1).
        """
        batch_size = embedded_title_input.size(0)
        embedded_text = self.embedded_text_input.expand(batch_size, -1, -1)

        _, (title_hidden, _) = self.model.title_encoder(embedded_title_input)
        _, (text_hidden, _) = self.model.text_encoder(embedded_text)

        title_repr = torch.cat((title_hidden[0], title_hidden[1]), dim=1)
        text_repr = torch.cat((text_hidden[0], text_hidden[1]), dim=1)

        combined = torch.cat((title_repr, text_repr), dim=1)
        x = self.model.dropout(combined)
        return torch.sigmoid(self.model.fc(x))


class WrapperTextBranchModel(nn.Module):
    """
    Wrapper model for analyzing the text branch of a dual-input model.

    This wrapper freezes the embedded title input and allows Captum or other
    attribution methods to analyze the text input exclusively.

    Args:
        model (nn.Module): Original dual-input model with separate encoders for title and text.
        embedded_title_input (torch.Tensor): Precomputed embedded tensor for the title input.
    """

    def __init__(self, model: nn.Module, embedded_title_input: torch.Tensor):
        super().__init__()
        self.model = model
        self.embedded_title_input = embedded_title_input

    def forward(self, embedded_text_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fixed title embedding and variable text input.

        Args:
            embedded_text_input (torch.Tensor): Embedded input tensor for the text branch, shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: Output predictions (probabilities), shape (batch_size, 1).
        """
        batch_size = embedded_text_input.size(0)
        embedded_title = self.embedded_title_input.expand(batch_size, -1, -1)

        _, (text_hidden, _) = self.model.text_encoder(embedded_text_input)
        _, (title_hidden, _) = self.model.title_encoder(embedded_title)

        text_repr = torch.cat((text_hidden[0], text_hidden[1]), dim=1)
        title_repr = torch.cat((title_hidden[0], title_hidden[1]), dim=1)

        combined = torch.cat((title_repr, text_repr), dim=1)
        x = self.model.dropout(combined)
        return torch.sigmoid(self.model.fc(x))
