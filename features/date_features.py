import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer that extracts cyclical and calendar-based features from a date column.

    The transformer generates sine/cosine transformations of the month and day-of-week,
    binary weekend indicator, and optionally scaled 'days until latest date' feature.

    Args:
        date_column (str, optional): Name of the column containing date values. Defaults to "date".
        scale_days (bool, optional): Whether to scale the 'days_until_max' feature. Defaults to False.
        scaler_type (str, optional): Type of scaler to use if scaling days ("standard" or "minmax"). Defaults to "standard".
    """

    def __init__(self, date_column="date", scale_days=False, scaler_type="standard"):
        self.date_column = date_column
        self.scale_days = scale_days
        self.scaler_type = scaler_type
        self.scaler = None
        self.columns_ = [
            "month_sin", "month_cos",
            "dow_sin", "dow_cos",
            "is_weekend",
            "days_until_max"
        ]

    def fit(self, X, y=None):
        """
        Fits the internal scaler (if enabled) on the 'days_until_max' feature.

        Args:
            X (pd.DataFrame): Input DataFrame containing the date column.
            y: Ignored. Present for compatibility with sklearn pipeline.

        Returns:
            DateFeatureExtractor: Fitted instance of the transformer.
        """
        X_ = self._prepare(X)
        if self.scale_days:
            if self.scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(X_[["days_until_max"]])
        return self

    def transform(self, X):
        """
        Transforms the input DataFrame into a NumPy array of extracted date features.

        Args:
            X (pd.DataFrame): Input DataFrame containing the date column.

        Returns:
            np.ndarray: Array of shape (n_samples, 6) with extracted date features.
        """
        X_ = self._prepare(X)
        if self.scale_days and self.scaler:
            X_["days_until_max"] = self.scaler.transform(X_[["days_until_max"]])
        return X_[self.columns_].values

    def _prepare(self, X):
        """
        Internal method to compute raw date features from the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with additional columns for all engineered features.
        """
        df = X.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df["month"] = df[self.date_column].dt.month
        df["dayofweek"] = df[self.date_column].dt.weekday

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["days_until_max"] = (df[self.date_column].max() - df[self.date_column]).dt.days

        return df
