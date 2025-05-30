import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Rosenbrock Function
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    import torch.optim
    from torch import tensor
    return np, plt, tensor, torch


@app.cell
def _():
    a = 1.0
    b = 1.0 # 1: easy; 100 : hard

    def f(x, y):
        return (a - x)**2 + b*(y-x**2)**2

    # Minimum: (a, a**2) = (1.0, 1.0), value 0.0
    return (f,)


@app.cell
def _(f, np, plt):
    def contour():
        x = np.linspace(-1.0, 3.0, 200)
        y = np.linspace(0.0, 2.0, 200)
    
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        cs = plt.contour(X, Y, Z, levels=50, colors="grey", linewidths=1.0)
        plt.plot(1,1, "k+",)
        #axis = plt.gca()
        #axis.clabel(cs, inline=True, fontsize=10)
        #plt.colorbar()
        plt.gca().set_aspect(1.0)
    
    contour()
    return (contour,)


@app.cell
def _(tensor):
    x = tensor(0.0, requires_grad=True)
    y = tensor(0.0, requires_grad=True)
    return x, y


@app.cell
def _(torch, x, y):
    optimizer = torch.optim.SGD([x, y], lr=0.1)
    return (optimizer,)


@app.cell
def _(f, optimizer, tensor, x, y):
    x.data = tensor(2.0)
    y.data = tensor(1.0)
    z = f(x, y)
    xs, ys, zs = [x.item()], [y.item()], [z.item()]
    for i in range(100):
        z = f(x, y)
        z.backward()
        optimizer.step()
        xs.append(x.item())
        ys.append(y.item())
        zs.append(z.item())
        x.grad.zero_()
        y.grad.zero_()
    print(xs[-1], ys[-1], zs[-1])
    return xs, ys


@app.cell
def _(contour, plt, xs, ys):
    contour()
    plt.plot(xs, ys, color="C0")
    _
    return


@app.cell
def _(np, xs, ys):
    xs_1 = np.array(xs)
    ys_1 = np.array(ys)
    error = np.sqrt((xs_1 - 1.0) ** 2 + (ys_1 - 1.0) ** 2)
    error
    return (error,)


@app.cell
def _(error, plt):
    plt.semilogy(error)
    plt.grid(True)
    None
    return


@app.cell
def _():
    # TODO: tweak the optimizer parameters, try the other optimizers.
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
