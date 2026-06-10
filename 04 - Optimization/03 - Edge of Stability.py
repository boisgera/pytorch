import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt


@app.cell
def _():
    sh      = 1.0      # Sharpness (of the quadratic function)
    lr_eos  = (2 / sh) # Learning rate at the edge of stability
    x0      = 1.0      # Starting point
    eps     = 1e-1
    return eps, lr_eos, sh, x0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Edge of Stability

    ## Quadratic Function

    Let $f_2$ be a a single variable $x$, with a strict minimum at $0$. We necessarily have for some positive $\lambda$

    $$
    f(x) = \frac{\lambda}{2} x^2.
    $$

    Now, we have the derivatives $f_2'(x) = \lambda x$
    and
    $f_2''(x) = \lambda$. For any quadratic function, the derivative of order 2, the **sharpness** of the function, is a constant.
    """)
    return


@app.cell
def _(sh):
    def f2(x):
        return 0.5 * sh * x * x

    def d_f2(x):
        return sh * x

    def d2_f2(x):
        # Make sure that the output is an array
        return sh * np.ones_like(x)

    return d2_f2, d_f2, f2


@app.cell(hide_code=True)
def _(f2):
    def _():
        xs = np.linspace(-2.0, 2.0, 1000)
        fig, ax = plt.subplots(1, 1)
        ax.plot(xs, f2(xs))
        ax.grid(True)
        ax.set_title("Graph of $f_2$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f_2(x)$")
        return mo.center(fig)
    _()
    return


@app.cell
def _(lr_eos, x0):
    def gd(d_f, lr=lr_eos, n=100, x0=x0):
        "Gradient Descent"
        x = x0
        xs = [x]
        for i in range(n):
            x = x - lr * d_f(x)
            xs.append(x)
        return np.array(xs)

    return (gd,)


@app.cell
def _(d2_f2, d_f2, f2, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f2, lr=lr_eos/4)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f2(xs))
        axes[1].set_ylabel("$f_2(x_n)$")
        axes[2].plot(d2_f2(xs))
        axes[2].set_ylabel("$f_2''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell
def _(d2_f2, d_f2, f2, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f2, lr=lr_eos/2)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f2(xs))
        axes[1].set_ylabel("$f_2(x_n)$")
        axes[2].plot(d2_f2(xs))
        axes[2].set_ylabel("$f_2''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell
def _(d2_f2, d_f2, f2, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f2, lr=lr_eos)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f2(xs))
        axes[1].set_ylabel("$f_2(x_n)$")
        axes[2].plot(d2_f2(xs))
        axes[2].set_ylabel("$f_2''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell
def _(d2_f2, d_f2, f2, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f2, lr=1.1*lr_eos)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f2(xs))
        axes[1].set_ylabel("$f_2(x_n)$")
        axes[2].plot(d2_f2(xs))
        axes[2].set_ylabel("$f_2''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cubic Function

    Consider a function $f_3$ with the same quadratic behavior as $f_2$ near $0$ but a (small) extra cubic term.

    $$
    f_3(x) = \frac{\lambda}{2} x^2 + \frac{\epsilon}{6} x^3.
    $$
    """)
    return


@app.cell
def _(eps, sh):
    def f3(x):
        return 0.5 * sh * x * x + (eps / 6) * x * x * x

    def d_f3(x):
        return sh * x + 0.5 * eps * x * x

    def d2_f3(x):
        return sh + eps * x

    return d2_f3, d_f3, f3


@app.cell
def _(f3):
    def _():
        xs = np.linspace(-2.0, 2.0, 1000)
        fig, ax = plt.subplots(1, 1)
        ax.plot(xs, f3(xs))
        ax.grid(True)
        ax.set_title("Graph of $f_3$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f_3(x)$")
        return mo.center(fig)
    _()
    return


@app.cell
def _(f3):
    def _():
        xs = np.linspace(-20.0, 20.0, 1000)
        fig, ax = plt.subplots(1, 1)
        ax.plot(xs, f3(xs))
        ax.grid(True)
        ax.set_title("Graph of $f_3$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f_3(x)$")
        return mo.center(fig)
    _()
    return


@app.cell
def _(d2_f3, d_f3, f3, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f3, lr=lr_eos/4)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f3(xs))
        axes[1].set_ylabel("$f_3(x_n)$")
        axes[2].plot(d2_f3(xs))
        axes[2].set_ylabel("$f_3''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell
def _(d2_f3, d_f3, f3, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f3, lr=lr_eos/2)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f3(xs))
        axes[1].set_ylabel("$f_3(x_n)$")
        axes[2].plot(d2_f3(xs))
        axes[2].set_ylabel("$f_3''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now, the gradient descent optimizer is still making progress (on average) at what was previously the edge of stability.
    """)
    return


@app.cell
def _(d2_f3, d_f3, f3, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f3, lr=lr_eos, n=100)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f3(xs))
        axes[1].set_ylabel("$f_3(x_n)$")
        axes[2].plot(d2_f3(xs))
        axes[2].set_ylabel("$f_3''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For some values of the learning rate that previously resulted in numerical blowups, we now still have bounded numerical values.
    """)
    return


@app.cell
def _(d2_f3, d_f3, f3, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f3, lr=1.1*lr_eos)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f3(xs))
        axes[1].set_ylabel("$f_3(x_n)$")
        axes[2].plot(d2_f3(xs))
        axes[2].set_ylabel("$f_3''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note that technically, the minimum is still (Lyapunov-)unstable: we do not stay close to the minimum even if we (very) close to the minimum.
    """)
    return


@app.cell
def _(d2_f3, d_f3, f3, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f3, lr=1.1*lr_eos, x0=1e-3)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f3(xs))
        axes[1].set_ylabel("$f_3(x_n)$")
        axes[2].plot(d2_f3(xs))
        axes[2].set_ylabel("$f_3''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note that for all this regime, the sharpness experimentally oscillates around the theoretical stability limit. There is some self-regulation going on!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If we still increase the learning rate, after a while, we reach a limit ...
    """)
    return


@app.cell
def _(d2_f3, d_f3, f3, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f3, lr=1.2*lr_eos)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f3(xs))
        axes[1].set_ylabel("$f_3(x_n)$")
        axes[2].plot(d2_f3(xs))
        axes[2].set_ylabel("$f_3''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell
def _(d2_f3, d_f3, f3, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f3, lr=1.5*lr_eos)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f3(xs))
        axes[1].set_ylabel("$f_3(x_n)$")
        axes[2].plot(d2_f3(xs))
        axes[2].set_ylabel("$f_3''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ... and afterwards, the blowup reappears.
    """)
    return


@app.cell
def _(d2_f3, d_f3, f3, gd, lr_eos):
    def _():
        fig, axes = plt.subplots(3, 1, sharex=True)
        xs = gd(d_f3, lr=1.501*lr_eos)
        axes[0].plot(xs)
        axes[0].set_ylabel("$x_n$")
        axes[1].plot(f3(xs))
        axes[1].set_ylabel("$f_3(x_n)$")
        axes[2].plot(d2_f3(xs))
        axes[2].set_ylabel("$f_3''(x_n)$")
        axes[2].set_xlabel("gradient descent iteration $n$")
        return mo.center(fig)
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Cobweb Analysis
    """)
    return


@app.function
def cobweb(f, x0, n=100, xlim=(0, 1)):
    xs = np.linspace(*xlim, 400)
    fig, ax = plt.subplots()
    ax.plot(xs, f(xs), "C0", lw=2)
    ax.plot(xs, xs, "--", color="gray")

    x, px, py = x0, [x0], [0.0]
    for _ in range(n):
        y = f(x)
        px += [x, x, x, y]
        py += [x, y, y, y]
        x = y

    ax.plot(px, py, "C1")
    ax.set(xlim=xlim, ylim=xlim, xlabel="$x_n$", ylabel="$x_{n+1}$", aspect="equal")
    return ax


@app.cell
def _(d_f3, lr_eos, x0):
    def _():
        lr = lr_eos * 1.1
        def gd_step(x):
            return x - lr * d_f3(x)
        return mo.center(cobweb(gd_step, x0=x0, xlim=(-20, 20)))
    _()
    return


if __name__ == "__main__":
    app.run()
