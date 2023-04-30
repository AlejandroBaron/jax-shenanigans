def MSE(B, X, y):
    """Computes Mean Squared Error"""
    se2 = (y - X @ B) ** 2 # quadratic errors
    return se2.mean()

def MSE_grad(w, X, y):
    "Explicit formulation for the gradient of the MSE"
    l = len(X)
    return (1/l) * 2 * (X.T @ (X @ w - y))