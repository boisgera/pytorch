import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    sh  = 1.0     # Sharpness (of the quadratic function)
    lr  = 2 / sh  # Learning rate at the edge of stability
    x0  = 1.0     # Starting point
    eps = 1e-1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// Warning | TODO

    - [ ] Automate the state + value + sharpness drawing
    - [ ] Explain/test that negative values of `eps` "work" too.
    - [ ] Comment, document
    - [ ] Test Fourth-Order with `eps = 0` ?
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Edge of Stability
    """)
    return


@app.cell
def _():
    def f2(x):
        return 0.5 * sh * x * x

    def d_f2(x):
        return sh * x

    def d2_f2(x):
        return sh * np.ones_like(x)

    return d2_f2, d_f2, f2


@app.function
def gd(d_f, lr=lr, n=100, x0=x0):
    x = x0
    xs = [x]
    for i in range(n):
        x = x - lr * d_f(x)
        xs.append(x)
    return np.array(xs)


@app.cell
def _(f2):
    _xs = np.linspace(-2.0, 2.0, 1000)
    plt.plot(_xs, f2(_xs))
    return


@app.cell
def _(d2_f2, d_f2, f2):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f2, lr=lr / 4)
    _axes[0].plot(_xs)
    _axes[1].plot(f2(_xs))
    _axes[2].plot(d2_f2(_xs))
    _fig
    return


@app.cell
def _(d2_f2, d_f2, f2):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f2, lr=lr / 2)
    _axes[0].plot(_xs)
    _axes[1].plot(f2(_xs))
    _axes[2].plot(d2_f2(_xs))
    _fig
    return


@app.cell
def _(d2_f2, d_f2, f2):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f2, lr=lr)
    _axes[0].plot(_xs)
    _axes[1].plot(f2(_xs))
    _axes[2].plot(d2_f2(_xs))
    _fig
    return


@app.cell
def _(d2_f2, d_f2, f2):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f2, lr=1.1*lr)
    _axes[0].plot(_xs)
    _axes[1].plot(f2(_xs))
    _axes[2].plot(d2_f2(_xs))
    _fig
    return


@app.cell
def _():
    def f3(x):
        return 0.5 * sh * x * x + (eps / 6) * x * x * x

    def d_f3(x):
        return sh * x + 0.5 * eps * x * x

    def d2_f3(x):
        return sh + eps * x

    return d2_f3, d_f3, f3


@app.cell
def _(f3):
    _xs = np.linspace(-2.0, 2.0, 1000)
    plt.plot(_xs, f3(_xs))
    return


@app.cell
def _(f3):
    _xs = np.linspace(-20.0, 20.0, 1000)
    plt.plot(_xs, f3(_xs))
    return


@app.cell
def _(d2_f3, d_f3, f3):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f3, lr=lr / 4)
    _axes[0].plot(_xs)
    _axes[1].plot(f3(_xs))
    _axes[2].plot(d2_f3(_xs))
    _fig
    return


@app.cell
def _(d2_f3, d_f3, f3):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f3, lr=lr / 2)
    _axes[0].plot(_xs)
    _axes[1].plot(f3(_xs))
    _axes[2].plot(d2_f3(_xs))
    _fig
    return


@app.cell
def _(d2_f3, d_f3, f3):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f3, lr=lr)
    _axes[0].plot(_xs)
    _axes[1].plot(f3(_xs))
    _axes[2].plot(d2_f3(_xs))
    _fig
    return


@app.cell
def _(d2_f3, d_f3, f3):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f3, lr=1.1*lr)
    _axes[0].plot(_xs)
    _axes[1].plot(f3(_xs))
    _axes[2].plot(d2_f3(_xs))
    _fig
    return


@app.cell
def _(d2_f3, d_f3, f3):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f3, lr=1.2*lr)
    _axes[0].plot(_xs)
    _axes[1].plot(f3(_xs))
    _axes[2].plot(d2_f3(_xs))
    _fig
    return


@app.cell
def _(d2_f3, d_f3, f3):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f3, lr=1.5*lr)
    _axes[0].plot(_xs)
    _axes[1].plot(f3(_xs))
    _axes[2].plot(d2_f3(_xs))
    _fig
    return


@app.cell
def _(d2_f3, d_f3, f3):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f3, lr=1.5001*lr)
    _axes[0].plot(_xs)
    _axes[1].plot(f3(_xs))
    _axes[2].plot(d2_f3(_xs))
    _fig
    return


@app.cell
def _(d2_f3, d_f3, f3):
    _fig, _axes = plt.subplots(3, 1, sharex=True)
    _xs = gd(d_f3, x0=1e-5, lr=1.5*lr)
    _axes[0].plot(_xs)
    _axes[1].plot(f3(_xs))
    _axes[2].plot(d2_f3(_xs))
    _fig
    return


if __name__ == "__main__":
    app.run()
