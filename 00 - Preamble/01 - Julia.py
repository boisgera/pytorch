import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    # Python Standard Library
    import time

    # Third-Party Libraries
    import torch
    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Julia Sets

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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
    """)
    return


@app.cell
def _():
    n = 100 # The computational load increases with n
    return (n,)


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # auto:
    torch.set_default_device(device)
    device
    return


@app.cell
def _():
    # Image dimensions
    width, height = 1600, 900

    # Real and imaginary limits for the complex plane
    re_min, re_max = -2.0, 2.0
    im_min, im_max = -1.5, 1.5
    return height, im_max, im_min, re_max, re_min, width


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Julia iteration:
    $$
    f(z) = z^2 - 0.8 + 0.156 i
    $$
    """)
    return


@app.cell
def _(height, im_max, im_min, re_max, re_min, width):
    def escape(n, radius=2.0):
        re = torch.linspace(re_min, re_max, width)
        im = torch.linspace(im_min, im_max, height)
        x, y = torch.meshgrid(re, im, indexing="xy")
        z = torch.complex(x, y)
        count = torch.zeros_like(x)
        for _ in range(n):
            z = z * z + (-0.8 + 0.156j)
            count += (z.abs() <= radius)
        return count

    return (escape,)


@app.cell
def _(escape, n):
    t_0 = time.perf_counter()
    julia_vals = escape(n)
    dt = time.perf_counter() - t_0
    return dt, julia_vals


@app.cell(hide_code=True)
def _(im_max, im_min, julia_vals, re_max, re_min):
    scale = 0.6
    plt.figure(figsize=(16*scale, 9*scale))
    plt.imshow(julia_vals.to("cpu"), cmap="hot", extent=[re_min, re_max, im_min, im_max])
    plt.axis("off")
    plt.tight_layout(pad=0)
    mo.center(plt.gcf())
    return


@app.cell(hide_code=True)
def _(dt, height, n, width):
    mo.md(rf"""
    /// note | Performance Analysis
    Each update of `z` and `count` require 14 floating-point operations. Since we have {width}$\times$ {height} = {width * height} points at each step, that makes {14 * width * height} floating points operations. The current number of steps `n` is {n}, thus we perform in total {14 * n * width * height} operations. Given that experimentally, that takes {dt:.3f} seconds, our throughput was {14 * n * width * height / dt:.1f} flops or approximately **{14 * n* width * height / dt / 1e9:.1f} GFLOPS**.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// warning

    On my laptop whose GPU is a NVIDIA Quadro P2000, I get a performance which is very small wrt the compute peak, which is **3.0 TFLOPS**.
    To explain why, we also need to analyze the memory bandwidth of the algorithm.


    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In single computation of

    ```python
    z = z * z + (-0.8 + 0.156j)
    count += (z.abs() <= radius)
    ```

    if there is no kernel fusion, then all intermediate values are transferred back and forth:

    |  # | Op kind | Expression                | Size (bytes) |
    | -: | ------- | ------------------------- | ------------ |
    |  1 | load    | `z`                       | 8            |
    |  2 | load    | `z`                       | 8            |
    |  3 | store   | `tmp_1 = z * z`           | 8            |
    |  4 | load    | `tmp_1`                   | 8            |
    |  5 | load    | `c = -0.8 + 0.156j`       | 8            |
    |  6 | store   | `tmp_1 + c`               | 8            |
    |  7 | load    | `z`                       | 8            |
    |  8 | store   | `tmp_2 = z.abs()`         | 4            |
    |  9 | load    | `tmp_2`                   | 4            |
    | 10 | load    | `radius`                  | 4            |
    | 11 | store   | `tmp_3 = tmp_2 <= radius` | 1            |
    | 12 | load    | `tmp_3`                   | 1            |
    | 13 | load    | `count`                   | 4            |
    | 14 | store   | `count + tmp_3`           | 4            |


    With a total of **76 bytes** moved at each step for exach pixel.
    """)
    return


@app.cell(hide_code=True)
def _(dt, height, n, width):
    mo.md(f"""
    That would lead to an estimate of **{76 * n * width * height / dt / 1e9:.2f} GB/s** memory bandwidth used. But the theoretical limit of my card is below that!

    | Feature | Value |
    |---|---|
    | Compute peak (FP32) | 3.0 TFLOPS |
    | Memory bandwidth | 140 GB/s |
    | VRAM | 5 GB |

    Therefore some of the data is probably cached. But anyway, since the computation rate seems well below the theoretical limit while the memory bandwidth is above the theoretical limit, we can conclude that the current process is memory-bound and not computation-bound.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""



    """)
    return


if __name__ == "__main__":
    app.run()
