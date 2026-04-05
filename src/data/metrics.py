"""Evaluation metrics utilities."""


def fpr_tpr(tp: int, fp: int, tn: int, fn: int):
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return {"fpr": fpr, "tpr": tpr}
