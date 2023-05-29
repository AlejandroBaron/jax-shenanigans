# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit, random

from jax_shenanigans.dl.losses import MSE, MSE_grad
from jax_shenanigans.utils.benchmarking import random_linear_setup, with_timing


@with_timing(return_t=True, log=False)
def gradient_descent(gradient_f, w, X, y, epochs: int = 60, lr: float = 7e-1):
    for _ in range(epochs):
        w = w - lr * gradient_f(w, X, y)
    return w


n = 100
p = 2  # number of parameters including linear bias
key = random.PRNGKey(0)
setup = random_linear_setup(n=n, p=p, key=key)


B, X, y = setup
B_np, X_np, y_np = map(np.asarray, setup)
B0 = random.uniform(key, (p, 1))
B0_np = np.asarray(B0)

times = {"raw": [], "jit": [], "np": []}

for _ in range(100):
    B_raw, t_raw = gradient_descent(grad(MSE), w=B0, X=X, y=y)
    B_jit, t_jit = gradient_descent(grad(jit(MSE)), B0, X=X, y=y)
    B_np, t_np = gradient_descent(MSE_grad, w=B0_np, X=X_np, y=y_np)

    times["raw"].append(t_raw)
    times["jit"].append(t_jit)
    times["np"].append(t_np)

# %%
plt.boxplot(list(times.values()))
plt.xticks([1, 2, 3], list(times.keys()))
plt.show()
# %%


def plot_B(b, **kwargs):
    x_plot = jnp.linspace(X.min(), X.max())
    plt.plot(x_plot, b[1] * x_plot + b[0], **kwargs)


plot_B(B, label="true")
plot_B(B_jit, label="adjusted jit")
plot_B(B_raw, label="adjusted raw", linestyle=":")
plt.scatter(X[:, 1], y, color="lightgreen", label="values")

plt.legend()
plt.show()

# %%
B_l, X_l, y_l = random_linear_setup(n**2, p, key=key)
B0_l = random.uniform(key, (p, 1))
B_l_hat, _ = gradient_descent(jit(MSE), B0_l)

plt.scatter(X_l[:, 1], y_l, color="lightgreen", label="values")
plot_B(B_l, label="real")
plot_B(B_l, label="adjusted jit")
plt.legend()
