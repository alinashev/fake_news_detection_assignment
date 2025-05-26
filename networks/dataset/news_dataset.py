from torch.utils.data import Dataset

class NewsDataset(Dataset):
    """
    PyTorch Dataset for fake news classification tasks.
    Each sample contains a title sequence, a text sequence, a label, and optionally date features.

    Args:
        title_seqs (list or np.ndarray): Tokenized sequences for news titles.
        text_seqs (list or np.ndarray): Tokenized sequences for news body texts.
        labels (list or np.ndarray): Binary labels (0 for real, 1 for fake).
        date_feats (list or np.ndarray, optional): Optional additional features (e.g., publication date features).
    """

    def __init__(self, title_seqs, text_seqs, labels, date_feats=None):
        self.title_seqs = title_seqs
        self.text_seqs = text_seqs
        self.labels = labels
        self.date_feats = date_feats

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with the following keys:
                - 'title': Title token sequence.
                - 'text': Text token sequence.
                - 'label': Corresponding label.
                - 'date' (optional): Date-related features if provided.
        """
        sample = {
            "title": self.title_seqs[idx],
            "text": self.text_seqs[idx],
            "label": self.labels[idx]
        }
        if self.date_feats is not None:
            sample["date"] = self.date_feats[idx]

        return sample
