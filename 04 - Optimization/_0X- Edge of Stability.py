import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rosenbrock "Banana" Function

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// warning | TODO

    - [ ] explain very low learning rate regime and limit curve (through ODE)
    - [ ] "classic" regime, explainable by quadratic stablity theory,
    - [ ] EOS (Edge of Stability) theory and self-regulation of sharpness.

    ///
    """)
    return


@app.cell
def _():
    import torch
    from torch import tensor
    from torch.func import grad, hessian, vmap
    import torch.optim

    return hessian, tensor, torch, vmap


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
        return (a - x) ** 2 + b * (y - x**2) ** 2

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
        plt.gcf()

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



    return (gradient_sequence,)


@app.cell
def _(contour, gradient_sequence, mo, plt, torch):
    xs0, ys0, zs0 = gradient_sequence(
        x_0=-0.25,
        y_0=1.75,
        n=10_000,
        Optimizer=torch.optim.SGD,
        lr=1e-2,
    )

    contour()
    plt.plot(xs0, ys0, "r-")
    plt.plot(xs0[-1], ys0[-1], "r+")
    mo.center(plt.gcf())
    return (zs0,)


@app.cell
def _(mo, plt, zs0):
    plt.semilogy(zs0)
    plt.ylabel("error")
    plt.xlabel("iteration")
    plt.grid(True)
    mo.center(plt.gcf())
    return


@app.cell
def _(gradient_sequence):
    xs, ys, zs = gradient_sequence(
        x_0=-0.25,
        y_0=1.75,
        n=1_000,
        lr=5e-3,
        momentum=0.92,
    )
    return xs, ys, zs


@app.cell
def _(contour, mo, plt, xs, ys):
    contour()
    plt.plot(xs, ys, "r-")
    plt.plot(xs[-1], ys[-1], "r+")
    mo.center(plt.gcf())
    return


@app.cell
def _(mo, plt, zs):
    plt.semilogy(zs)
    plt.ylabel("error")
    plt.xlabel("iteration")
    plt.grid(True)
    mo.center(plt.gcf())
    return


@app.cell
def _(contour, gradient_sequence, mo, plt, torch):
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
    mo.center(plt.gcf())
    return (zs2,)


@app.cell
def _(mo, plt, zs2):
    plt.semilogy(zs2)
    plt.ylabel("error")
    plt.xlabel("iteration")
    plt.grid(True)
    plt.gcf()
    mo.center(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sharpness
    """)
    return


@app.cell
def _(f, hessian, plt, torch, vmap):
    def plot_sharpness(lr=None):
        H = hessian(f)
        x = torch.linspace(-2.0, 3.0, 200)
        y = torch.linspace(-1.0, 3.0, 200)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        def fvec(xy):
            return f(xy[0], xy[1])

        def sharpness_at(xy):
            H = hessian(fvec)(xy)  # (2,2) at a single point
            return torch.linalg.eigvalsh(H)[-1]  # scalar

        points = torch.stack([X.ravel(), Y.ravel()], dim=1)  # (40000, 2)
        Z = vmap(sharpness_at)(points).reshape(200, 200)

        cs = plt.contour(
            X.numpy(),
            Y.numpy(),
            Z.detach().numpy(),
            levels=[2.0 / lr],
            colors="k",
            linestyles="dashed",
        )
        plt.clabel(cs, inline=True, fontsize=8)

        cs = plt.contour(
            X.numpy(),
            Y.numpy(),
            Z.detach().numpy(),
            levels=[50, 100, 200, 400],
            colors="black",
        )
        plt.clabel(cs, inline=True, fontsize=8)

        plt.plot(1, 1, "k+")
        plt.grid(True)
        plt.gca().set_aspect(1.0)
        return plt.gcf()

    return (plot_sharpness,)


@app.cell
def _(gradient_sequence, mo, plot_sharpness, plt, torch):
    def _(lr=1e-3):
        print(f"max sharpness: {2/lr}")
        plot_sharpness(lr=lr)
        xs3, ys3, zs3 = gradient_sequence(
            x_0=-0.25,
            y_0=1.75,
            n=10_000,
            Optimizer=torch.optim.SGD,
            lr=lr,
        )
        plt.plot(xs3, ys3, "r-")
        plt.plot(xs3[-1], ys3[-1], "r+")
        return mo.center(plt.gcf())

    # lr=1e-2 # the "good", classicaley suggested value.
    # lr=2e-2 # ~ the theoretical limit
    _(lr = 2e-2) # here, we have a max sharpness of 50
    # and would be divergent in the quadratic model.
    # There are some non-quadratic effect that maintain us -- for a while! --
    # in the boundary of sharpness = 50
    return


@app.cell
def _(gradient_sequence, mo, plt, torch):
    def _(lr=1e-2):
        print(f"max sharpness: {2/lr}")
        xs, ys, zs = gradient_sequence(
            x_0=-0.25,
            y_0=1.75,
            n=10_000,
            Optimizer=torch.optim.SGD,
            lr=lr,
        )
        plt.semilogy(zs)
        plt.ylabel("error")
        plt.xlabel("iteration")
        plt.grid(True)
        return mo.center(plt.gcf())

    # Interesting : 1e-3 (too slow), 1e-2 (classic recom.), 2e-2 (limit), 3e-2 and 4e-2 (EOS mode)    
    _(lr=3e-2)
    return


@app.cell
def _(f, gradient_sequence, hessian, mo, plt, torch, vmap):
    def _(lr=1e-2):
        print(f"max sharpness: {2/lr}")
        xs, ys, _ = gradient_sequence(
            x_0=-0.25,
            y_0=1.75,
            n=10_000,
            Optimizer=torch.optim.SGD,
            lr=lr,
        )
        def fvec(xy):
            return f(xy[0], xy[1])

        def sharpness_at(xy):
            H = hessian(fvec)(xy)  # (2,2) at a single point
            return torch.linalg.eigvalsh(H)[-1]  # scalar

        points = torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1)  # (40000, 2)
        zs = vmap(sharpness_at)(points)


        plt.plot(zs, alpha=0.5, label="sharpness")
        plt.plot(2.0 / lr * torch.ones_like(zs), label="max sharpness")
        plt.ylabel("sharpness")
        plt.xlabel("iteration")
        plt.grid(True)
        plt.legend()
        return mo.center(plt.gcf())

    # Interesting : 1e-3 (too slow), 1e-2 (classic recom.), 2e-2 (limit), 3e-2 and 4e-2 (EOS mode)    
    _(lr=4e-2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
