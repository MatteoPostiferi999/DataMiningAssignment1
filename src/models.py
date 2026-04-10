import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    classification_report,
)


def evaluate_regression(y_true, y_pred) -> dict:
    """Return MSE, RMSE and MAE for a regression prediction."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def evaluate_classification(y_true, y_pred) -> dict:
    """Return accuracy and macro F1 for a classification prediction."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro"),
        "Report": classification_report(y_true, y_pred),
    }
