from .utils.typing import array

def MSE(w: array, X:array, y:array):
    """Computes Mean Squared Error"""
    se2 = (y - X @ w) ** 2 # quadratic errors
    return se2.mean()

def MSE_grad(w:array, X:array, y:array):
    "Explicit formulation for the gradient of the MSE"
    l = len(X)
    return (1/l) * 2 * (X.T @ (X @ w - y))