import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Rosenbrock "Banana" Function

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell
def _():
    import torch
    from torch import tensor
    import torch.optim
    return tensor, torch


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _():
    a = 1.0
    b = 10.0 # b = 1.0: easy; b = 100.0: hard

    def f(x, y):
        return (a - x) ** 2 + b *(y - x**2)**2

    # Minimum: (a, a**2) = (1.0, 1.0), value 0.0
    return (f,)


@app.cell
def _(f, plt, torch):
    def contour():
        x = torch.linspace(-1.0, 3.0, 200)
        y = torch.linspace(-1.0, 3.0, 200)

        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = f(X, Y)
        cs = plt.contour(X, Y, Z, levels=torch.linspace(0.0, 10.0, 21))
        plt.plot(1,1, "k+",)
        plt.colorbar()
        plt.grid(True)
        plt.gca().set_aspect(1.0)
        return plt.gcf()

    contour()
    return (contour,)


@app.cell
def _(f, tensor, torch):
    def gradient_sequence(
        x_0=0.0,
        y_0=0.0,
        n=100,
        Optimizer=torch.optim.SGD,
        **options,
    ):
        x = tensor(x_0, requires_grad=True)
        y = tensor(y_0, requires_grad=True)
        optimizer = Optimizer([x, y], **options)

        xs, ys, zs = [], [], []
        for i in range(n):
            z = f(x, y)
            xs.append(x.item())
            ys.append(y.item())
            zs.append(z.item())

            z.backward()
            optimizer.step()
            x.grad = None
            y.grad = None

        return xs, ys, zs


    xs, ys, zs = gradient_sequence(
        x_0=-0.25,
        y_0=1.75,
        n=1_000,
        lr=5e-3,
        momentum=0.92,
    )
    return gradient_sequence, xs, ys, zs


@app.cell
def _(contour, plt, xs, ys):
    contour()
    plt.plot(xs, ys, "r-")
    plt.plot(xs[-1], ys[-1], "r+")
    return


@app.cell
def _(plt, zs):
    plt.semilogy(zs)
    plt.ylabel("error")
    plt.xlabel("iteration")
    plt.grid(True)
    plt.gcf()
    return


@app.cell
def _(contour, gradient_sequence, plt, torch):
    xs2, ys2, zs2 = gradient_sequence(
        x_0=-0.25,
        y_0=1.75,
        n=1_000,
        Optimizer=torch.optim.Adam,
        lr=6e-1,
    )

    contour()
    plt.plot(xs2, ys2, "r-")
    plt.plot(xs2[-1], ys2[-1], "r+")
    return (zs2,)


@app.cell
def _(plt, zs2):
    plt.semilogy(zs2)
    plt.ylabel("error")
    plt.xlabel("iteration")
    plt.grid(True)
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
