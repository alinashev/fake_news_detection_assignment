import torch
from captum.attr import IntegratedGradients

class AttributionExplainer:
    """
    Explainer class that uses Integrated Gradients to compute feature attributions
    for a given PyTorch model and its interpretable embedding.

    Attributes:
        model (torch.nn.Module): The model to interpret.
        interpretable_emb: Interpretable embedding wrapper or layer for Captum.
    """

    def __init__(self, model: torch.nn.Module, interpretable_emb):
        """
        Initializes the AttributionExplainer.

        Args:
            model (torch.nn.Module): The model for which attributions will be computed.
            interpretable_emb: The interpretable embedding layer used by Captum.
        """
        self.model = model
        self.interpretable_emb = interpretable_emb

    def compute_attributions(self, input_embedded: torch.Tensor, target: int = 0, n_steps: int = 50) -> torch.Tensor:
        """
        Computes feature attributions using the Integrated Gradients method.

        Args:
            input_embedded (torch.Tensor): Embedded input tensor of shape (1, seq_len, embedding_dim).
            target (int, optional): The target class index for which attribution is computed. Defaults to 0.
            n_steps (int, optional): The number of steps for the Riemann approximation of the integral. Defaults to 50.

        Returns:
            torch.Tensor: Attribution scores for each input feature of shape (seq_len, embedding_dim).
        """
        ig = IntegratedGradients(self.model)
        with torch.backends.cudnn.flags(enabled=False):
            attributions = ig.attribute(
                inputs=input_embedded,
                baselines=torch.zeros_like(input_embedded),
                target=target,
                n_steps=n_steps
            )
        return attributions.squeeze(0).detach().cpu()
