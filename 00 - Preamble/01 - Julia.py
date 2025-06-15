import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Julia Sets

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Learning Objectives

     - [ ] Get a gentle introduction to Julia fractals,
     - [ ] Discover what tensor/vectorized computations are about in this context,
     - [ ] Get an estimate of the number of floating-points operations you require,
     - [ ] Manage to install a Pytorch version with CUDA enabled,
     - [ ] Have an example of the Pytorch API for device selection (CPU or GPU),
     - [ ] See what difference the device can make in computation time or FLOPS,
     - [ ] See what difference the tensor data type makes in computation time of FLOPS,
     - [ ] Discover the `top` and `nvdia-smi` utilities for CPU/GPU activity monitoring.

    ///
    """
    )
    return


@app.cell
def _():
    n = 100 # The computational load increases with n
    return (n,)


@app.cell
def _(torch):
    dtype = torch.float
    return (dtype,)


@app.cell
def _(escape, n, time):
    t_0 = time.perf_counter()
    julia_vals = escape(n)
    dt = time.perf_counter() - t_0
    return dt, julia_vals


@app.cell(hide_code=True)
def _(im_max, im_min, julia_vals, mo, plt, re_max, re_min):
    scale = 0.6
    plt.figure(figsize=(16*scale, 9*scale))
    plt.imshow(julia_vals.to("cpu"), cmap="hot", extent=[re_min, re_max, im_min, im_max])
    plt.axis("off")
    plt.tight_layout(pad=0)
    mo.center(plt.gcf())
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import time
    return (time,)


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import torch
    return plt, torch


@app.cell
def _(torch):
    device = torch.device("cpu") # auto: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return (device,)


@app.cell(hide_code=True)
def _():
    # Image dimensions
    width, height = 1600, 900

    # Real and imaginary range for the complex plan
    re_min, re_max = -2.0, 2.0
    im_min, im_max = -1.5, 1.5

    # Maximum iterations for escape time
    max_iter = 1000
    return height, im_max, im_min, re_max, re_min, width


@app.cell
def _(device, dtype, height, im_max, im_min, re_max, re_min, torch, width):
    c = -0.8 + 0.156j

    def f(z):
        return z * z + c

    def escape(max_iter=1_000, radius=2.0):
        "Round n at which |f^n(z)| > radius"
        re = torch.linspace(re_min, re_max, width, dtype=dtype, device=device)
        im = torch.linspace(im_min, im_max, height, dtype=dtype, device=device)
        x, y = torch.meshgrid(re, im, indexing="xy")
        z = torch.complex(x, y)
        count = torch.zeros_like(z, dtype=dtype)
        mask = (z.abs() <= radius)
        for i in range(max_iter):
            z = f(z)
            mask = (z.abs() <= radius)
            count[mask] += 1
        return count
    return (escape,)


@app.cell(hide_code=True)
def _(dt, height, mo, n, width):
    mo.md(rf"""The computation of `f(z)` for a single `z` requires 8 floating-operations. Since we compute `f(z)` for {width}$\times$ {height} = {width * height} points at each steps, that makes {8 * width * height} floating points operations. The current number of steps `n` is {n}, thus we perform in total {8 * n * width * height} operations. Given that experimentally, that takes {dt:.3f} seconds, our throughput was {8 * n * width * height / dt:.1f} flops or approximately {8 * n* width * height / dt / 1e9:.3f} Gflops.""")
    return


if __name__ == "__main__":
    app.run()
