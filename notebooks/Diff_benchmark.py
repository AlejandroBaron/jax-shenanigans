# %%
from itertools import product
from random import choices, seed

import matplotlib.pyplot as plt
from jax import grad, jacfwd, jacrev, jit, random, vmap
from jaxlib.xla_extension import XlaRuntimeError
from loguru import logger
from numpy import median, nan
from tqdm import tqdm

from jax_shenanigans.utils.benchmarking import with_timing

seed(123456)


@jit
def forward(W, X):
    return X @ W


@with_timing(return_t=True, log=False)
def test_jacfwd(W, X):
    return jacfwd(forward)(W, X).sum(axis=1)


@with_timing(return_t=True, log=False)
def test_vmap_grad(W, X):
    return vmap(
        lambda xi: vmap(lambda wi: grad(forward)(wi, xi), in_axes=-1, out_axes=-1)(W)
    )(X)


@with_timing(return_t=True, log=False)
def test_vmap_jacfwd(W, X):
    return vmap(lambda xi: jacfwd(forward)(W, xi))(X).sum(axis=1)


@with_timing(return_t=True, log=False)
def test_jacrev(W, X):
    return jacrev(forward)(W, X).sum(axis=1)


@with_timing(return_t=True, log=False)
def test_vmap_jacrev(W, X):
    return vmap(lambda xi: jacrev(forward)(W, xi))(X).sum(axis=1)


tests = [test_jacfwd, test_vmap_grad, test_vmap_jacfwd, test_jacrev, test_vmap_jacrev]
tests = {t.__name__: t for t in tests}


# %%


def test_loop(
    ns: list[int],
    ps: list[int],
    os: list[int],
    nreps: int = 200,
    tests: dict[str] = tests,
):
    setups = list(product(ns, ps, os))
    avg_times = {setup: {} for setup in setups}
    for n, p, o in tqdm(setups):
        xkey, wkey = random.PRNGKey(0), random.PRNGKey(1)
        logger.info(f"n={n}, p={p}")
        X = random.uniform(xkey, (n, p))
        W = random.uniform(wkey, (p, o))

        for tname, test in tests.items():
            test_times = []
            for _ in tqdm(range(nreps), desc=f"Running simulations for {tname}..."):
                try:
                    _, t = test(W, X)
                except XlaRuntimeError:
                    t = nan
                test_times.append(t)
            avg_times[(n, p, o)][tname] = median(test_times)

    return avg_times


test_pallete = {test: "#" + "".join(choices("0123456789ABCDEF", k=6)) for test in tests}


def plot_avg_times(avg_times, tests: dict = tests):
    ns = sorted({n for n, _, _ in avg_times})
    ps = sorted({p for _, p, _ in avg_times})
    os = sorted({o for _, _, o in avg_times})

    n_plots = len(os) * len(ns)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 10))
    if n_plots == 1:
        axs = [axs]
    else:
        axs.flatten()

    for o_n, ax in zip(product(os, ns), axs):
        o, n = o_n
        for test in sorted(tests):
            times = [avg_times[(n, p, o)][test] for p in ps]
            ax.plot(times, label=test, color=test_pallete[test])
            ax.set_xticks(range(len(ps)), ps)
            ax.set_title(f"o = {o}, n = {n}")
            ax.set_xlabel("p")
            ax.set_yscale("log")
            # ax.legend()
    lines, labels = ax.get_legend_handles_labels()
    labels = [label.replace("test_", "") for label in labels]
    fig.legend(lines, labels, loc="lower center", ncol=4)


# %%
tall_times = test_loop(ns=[50, 100], ps=[16, 128, 512], os=[2, 5])
# %%
plot_avg_times(tall_times)
plot_avg_times(tall_times, tests={t for t in tests if t != "test_jacrev"})
# %%

long_times = test_loop(ns=[50, 100], ps=[2, 16], os=[32, 64, 128, 256])

plot_avg_times(long_times)
plot_avg_times(long_times, tests={t for t in tests if t != "test_jacrev"})
