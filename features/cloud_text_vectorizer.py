from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

class CountTextVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom text vectorizer that applies CountVectorizer to a specific text column in a DataFrame.

    This class is designed for use in sklearn pipelines and supports optional tokenization
    and configuration of vocabulary size and frequency filtering.

    Args:
        text_column (str): Name of the column containing text data.
        tokenizer (callable, optional): Custom tokenizer or preprocessing function. Defaults to None.
        max_features (int, optional): Maximum number of features (vocabulary size). Defaults to 1000.
        min_df (int, optional): Minimum number of documents a term must appear in. Defaults to 2.
    """

    def __init__(self, text_column="full_text", tokenizer=None, max_features=1000, min_df=2):
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer = CountVectorizer(
            tokenizer=self.tokenizer,
            max_features=self.max_features,
            min_df=self.min_df,
        )

    def fit(self, X):
        """
        Fits the internal CountVectorizer on the specified text column of the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame containing the text column.

        Returns:
            CountTextVectorizer: The fitted instance of the transformer.
        """
        texts = X[self.text_column].astype(str).tolist()
        self.vectorizer.fit(texts)
        return self

    def transform(self, X):
        """
        Transforms the text column of the input DataFrame into a sparse matrix of token counts.

        Args:
            X (pd.DataFrame): Input DataFrame containing the text column.

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix of shape (n_samples, n_features).
        """
        texts = X[self.text_column].astype(str).tolist()
        return self.vectorizer.transform(texts)
