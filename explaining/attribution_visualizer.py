import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

class AttributionVisualizer:
    """
    A class for visualizing feature attributions from NLP models using different styles:
    colored text, bar charts, and heatmaps.

    Attributes:
        filter_tokens (list[str]): Tokens to exclude from visualization (e.g., "<PAD>").
        normalize (bool): Whether to normalize attributions between 0 and 1.
    """

    def __init__(self, filter_tokens=None, normalize=True):
        """
        Initializes the AttributionVisualizer.

        Args:
            filter_tokens (list[str], optional): Tokens to filter out. Defaults to ["<PAD>"].
            normalize (bool, optional): Whether to normalize attributions. Defaults to True.
        """
        self.filter_tokens = filter_tokens or ["<PAD>"]
        self.normalize = normalize

    def _filter_and_normalize(self, words, attributions):
        """
        Filters out unwanted tokens and optionally normalizes attributions.

        Args:
            words (list[str]): List of tokens (words).
            attributions (list[float]): Corresponding attribution scores.

        Returns:
            Tuple[list[str], np.ndarray]: Filtered words and processed attribution scores.
        """
        filtered = [(w, a) for w, a in zip(words, attributions) if w not in self.filter_tokens]
        if not filtered:
            return [], []

        words, attributions = zip(*filtered)
        attributions = np.array(attributions)
        if self.normalize:
            attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
        return list(words), attributions

    def show_colored_text(self, words, attributions, title="", max_words=20, cmap="Reds"):
        """
        Displays colored text highlighting token importance based on attribution scores.

        Args:
            words (list[str]): List of tokens to display.
            attributions (list[float]): Corresponding attribution scores.
            title (str, optional): Title to display above the text. Defaults to "".
            max_words (int, optional): Maximum number of tokens to show. Defaults to 20.
            cmap (str, optional): Name of the matplotlib colormap. Defaults to "Reds".
        """
        words, attributions = self._filter_and_normalize(words, attributions)
        if not words:
            print("No tokens to display.")
            return

        colormap = plt.colormaps.get_cmap(cmap)
        html = ""
        for word, score in zip(words[:max_words], attributions[:max_words]):
            rgba = colormap(score)
            color = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]:.2f})"
            html += f"<span style='background-color:{color}; padding:2px; margin:1px'>{word}</span> "

        display(HTML(f"<h4>{title}</h4>"))
        display(HTML(f"<div style='font-family:monospace; line-height: 2;'>{html}</div>"))

    def plot_bar_chart(self, words, attributions, title="", top_k=20):
        """
        Plots a horizontal bar chart of the top-k most important tokens.

        Args:
            words (list[str]): List of tokens.
            attributions (list[float]): Corresponding attribution scores.
            title (str, optional): Plot title. Defaults to "".
            top_k (int, optional): Number of top tokens to show. Defaults to 20.
        """
        words, attributions = self._filter_and_normalize(words, attributions)
        if not words:
            return

        top_idx = np.argsort(attributions)[-top_k:][::-1]
        top_words = [words[i] for i in top_idx]
        top_scores = attributions[top_idx]

        height_per_word = 0.4
        plt.figure(figsize=(10, max(4, top_k * height_per_word)))

        sns.barplot(x=top_scores, y=top_words, hue=top_words, palette="Reds", legend=False)
        plt.title(title, fontsize=14)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis="x", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, words, attributions, title="", max_len=60):
        """
        Plots a heatmap of attributions for a limited number of tokens.

        Args:
            words (list[str]): List of tokens.
            attributions (list[float]): Corresponding attribution scores.
            title (str, optional): Heatmap title. Defaults to "".
            max_len (int, optional): Maximum number of tokens to show. Defaults to 60.
        """
        words, attributions = self._filter_and_normalize(words, attributions)
        if not words or len(words) > max_len:
            return

        plt.figure(figsize=(2.5, max(4, len(words) * 0.35)))
        sns.heatmap(
            np.array(attributions).reshape(-1, 1),
            annot=np.array(words).reshape(-1, 1),
            fmt="",
            cmap="Reds",
            xticklabels=[title],
            yticklabels=words,
            cbar=True
        )
        plt.title(f"Heatmap — {title}")
        plt.tight_layout()
        plt.show()
