import numpy as np


def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).

    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    supported_averages = {"micro", "macro", "weighted", "binary"}
    if average not in supported_averages:
        raise ValueError(f"average must be one of {sorted(supported_averages)}")

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / y_true.size

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    labels = np.unique(np.concatenate((y_true, y_pred)))
    accuracy = np.mean(y_true == y_pred)

    if average == "micro":
        return {
            "accuracy": float(accuracy),
            "precision": float(accuracy),
            "recall": float(accuracy),
            "f1": float(accuracy),
        }

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        support = np.sum(y_true == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)

    if average == "macro":
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1_scores)
    else:
        weights = np.asarray(supports, dtype=float)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = np.average(precisions, weights=weights)
            recall = np.average(recalls, weights=weights)
            f1 = np.average(f1_scores, weights=weights)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }