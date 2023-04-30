# %%
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax_shenanigans.utils import with_timing
import matplotlib.pyplot as plt

n = 100
p = 2  # number of parameters including linear bias
key = random.PRNGKey(0)

def random_setup(n, p):
    X = random.uniform(key, (n, p - 1))
    X = jnp.concatenate([jnp.ones((n, 1)), X], axis=1)
    B = random.randint(key, (p, 1), 0, 10)
    _ , e_key = random.split(key)
    epsilon = random.normal(e_key, (n, 1))*0.3
    y = X.dot(B) + epsilon
    return B, X, y

B, X, y = random_setup(n,p)

# Internally, random.normal uses a uniform generator. If
# the same key is used, the correlation with a normal
# distribution is almost 1




# %%

def MSE(B, X, y):
    se2 = (y - X.dot(B)) ** 2 # quadratic errors
    return se2.mean()

@with_timing(return_t=True, log=False)
def gradient_descent(
    loss, 
    w,
    X=X,
    y=y,
    epochs: int = 60, 
    lr: float = 7e-1
):
    for _ in range(epochs):
        w = w - lr * grad(loss)(w, X, y)
    return w


B0 = random.uniform(key, (p, 1))

times = {
    'raw':[],
    'jit':[]
}
for _ in range(100):
    B_raw, t_raw = gradient_descent(MSE, B0)
    B_jit, t_jit = gradient_descent(jit(MSE), B0)

    times['raw'].append(t_raw)
    times['jit'].append(t_jit)

# %%
plt.boxplot(list(times.values()))
plt.xticks([1,2], list(times.keys()))
# %%

def plot_B(b, **kwargs):
    x_plot = jnp.linspace(X.min(), X.max())
    plt.plot(x_plot, b[1]*x_plot+b[0], **kwargs)


plot_B(B, label='true')
plot_B(B_jit, label='adjusted jit')
plot_B(B_raw, label='adjusted raw', linestyle=':')
plt.scatter(X[:,1], y, color='lightgreen', label='values')

plt.legend()
plt.show()

# %%
B_l, X_l, y_l = random_setup(n**2,p)
B0_l = random.uniform(key, (p, 1))
B_l_hat, _ = gradient_descent(jit(MSE), B0_l)

plt.scatter(X_l[:,1], y_l, color='lightgreen', label='values')
plot_B(B_l, label='real')
plot_B(B_l, label='adjusted jit')
plt.legend()