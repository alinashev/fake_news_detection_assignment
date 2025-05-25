import seaborn as sns
import matplotlib.pyplot as plt


class ModelErrorAnalyzer:
    """Class for analyzing classification model errors.

    This class provides tools to identify and analyze misclassified validation samples
    (false positives and false negatives). It includes functionality to visualize the
    probability distributions of these errors and highlight the features that contribute
    most to the misclassifications.

    Attributes:
        model: A trained classification model with attributes `val_prediction` (predicted labels)
            and `val_pr_proba` (predicted probabilities for the positive class).
        X_val (pd.DataFrame): Feature matrix for the validation set.
        y_val (pd.Series or np.array): True target labels for the validation set.
        y_pred (np.array): Predicted class labels.
        y_pred_proba (np.array): Predicted probabilities for the positive class.
    """

    def __init__(self, model, X_val, y_val):
        """Initializes the ModelErrorAnalyzer with a trained model and validation data.

        Args:
            model: Trained classification model with prediction and probability attributes.
            X_val (pd.DataFrame): Feature matrix for the validation set.
            y_val (pd.Series or np.array): True labels for the validation set.
        """
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.y_pred = model.val_prediction
        self.y_pred_proba = model.val_pr_proba

    def get_misclassified_samples(self):
        """Identifies false positives and false negatives from the validation set.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing false positives and
            false negatives, respectively. Each includes real labels, predicted labels,
            and predicted probabilities.
        """
        false_positives = self.X_val[(self.y_val == 0) & (self.y_pred == 1)].copy()
        false_negatives = self.X_val[(self.y_val == 1) & (self.y_pred == 0)].copy()

        false_positives["y_real"] = 0
        false_positives["y_pred"] = 1
        false_positives["y_pred_proba"] = self.y_pred_proba[(self.y_val == 0) & (self.y_pred == 1)]

        false_negatives["y_real"] = 1
        false_negatives["y_pred"] = 0
        false_negatives["y_pred_proba"] = self.y_pred_proba[(self.y_val == 1) & (self.y_pred == 0)]

        return false_positives, false_negatives

    def visualize_misclassified_samples(self):
        """Visualizes the distribution of predicted probabilities for misclassified samples.

        Generates histograms for false positives and false negatives showing the
        distribution of their predicted probabilities.
        """
        false_positives, false_negatives = self.get_misclassified_samples()

        plt.figure(figsize=(12, 5))
        sns.histplot(false_positives["y_pred_proba"], bins=20, kde=True, color='blue', label="False Positives")
        sns.histplot(false_negatives["y_pred_proba"], bins=20, kde=True, color='red', label="False Negatives",
                     alpha=0.7)
        plt.xlabel("Prediction Probability")
        plt.ylabel("Number of Samples")
        plt.title("Probability Distribution of Misclassified Samples")
        plt.legend()
        plt.show()

    def analyze_top_features(self, top_n=10):
        """Analyzes features that contribute most to misclassifications.

        Computes the mean absolute deviation of features in false positives and false negatives
        from the overall validation set means, and visualizes the top deviating features.

        Args:
            top_n (int): Number of top features to display for each type of error. Default is 10.
        """
        false_positives, false_negatives = self.get_misclassified_samples()
        df_val_means = self.X_val.mean()

        fp_diff = (false_positives.mean() - df_val_means).abs().sort_values(ascending=False).head(top_n)
        fn_diff = (false_negatives.mean() - df_val_means).abs().sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(y=fp_diff.index, x=fp_diff.values, color="blue", label="False Positives")
        sns.barplot(y=fn_diff.index, x=fn_diff.values, color="red", label="False Negatives", alpha=0.7)
        plt.xlabel("Deviation from Mean")
        plt.ylabel("Features")
        plt.title("Top Features Contributing to Misclassifications")
        plt.legend()
        plt.show()
