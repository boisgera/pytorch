import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Descente de Gradient et Fonction Quadratique
        """
    )
    return


@app.cell
def _():
    import torch 
    import matplotlib.pyplot as plt
    return plt, torch


@app.function
def f(x1, x2):
    return x1*x1 -0.5*x1*x2 - 0.5*x2*x1 + 3*x2*x2 + 2.0*x1 + 3.0*x2 + 1.0


@app.cell
def _(torch):
    x = torch.linspace(-5.0, 5.0, 100)
    _y = torch.linspace(-5.0, 5.0, 100)
    X, Y = torch.meshgrid(x, _y, indexing='xy')
    Z = f(X, Y)
    return X, Y, Z


@app.cell
def _(X, Y, Z, plt):
    _cs = plt.contour(X, Y, Z)
    plt.grid(True)
    plt.clabel(_cs)
    None
    return


@app.cell
def _(torch):
    A = torch.tensor([[1.0, -0.5], [-0.5, 3]])
    b = torch.tensor([2.0, 3.0])
    c = torch.tensor(1.0)

    def f_alt(x): # x: tensor, shape: (2,), ou shape (n, 2), ou shape (m, n, 2)
        xAx = torch.einsum("...i,ij,...j->...", x, A, x)
        bx = torch.einsum("i,...i->...", b, x)
        return xAx + bx + c
    return (f_alt,)


@app.cell
def _(f_alt, torch):
    x_1 = torch.tensor([1.0, 2.0])
    print(f(x_1[0], x_1[1]))
    print(f_alt(x_1))
    return


@app.cell
def _(torch):
    x_2 = torch.tensor([[1.0, 2.0], [-1.0, 1.0], [1.0, 2.0], [-5.0, -5.0]])
    f(x_2[:, 0], x_2[:, 1])
    return (x_2,)


@app.cell
def _(f_alt, x_2):
    f_alt(x_2)
    return


@app.cell
def _(torch):
    x_3 = torch.tensor([[[1.0, 2.0], [-1.0, 1.0]], [[1.0, 2.0], [-5.0, -5.0]]])
    f(x_3[:, :, 0], x_3[:, :, 1])
    return (x_3,)


@app.cell
def _(f_alt, x_3):
    f_alt(x_3)
    return


@app.cell
def _(f_alt, torch):
    x_4 = torch.linspace(-5.0, 5.0, 100)
    _y = torch.linspace(-5.0, 5.0, 100)
    X_1, Y_1 = torch.meshgrid(x_4, _y, indexing='xy')
    _XY = torch.stack((X_1, Y_1), dim=2)
    print(_XY.shape)
    Z_1 = f_alt(_XY)
    return X_1, Y_1, Z_1


@app.cell
def _(X_1, Y_1, Z_1, plt):
    _cs = plt.contour(X_1, Y_1, Z_1)
    plt.grid(True)
    plt.clabel(_cs)
    None
    return


@app.cell
def _(X_1, Y_1, Z_1, f_alt, plt, torch):
    x_5 = torch.tensor([4.0, 4.0], requires_grad=True)
    xs = [x_5.detach().clone()]
    _lr = 0.3
    _n = 100
    _optimizer = torch.optim.SGD(params=[x_5], lr=_lr)
    for _i in range(_n):
        _y = f_alt(x_5)
        _y.backward()
        _optimizer.step()
        xs.append(x_5.detach().clone())
        _optimizer.zero_grad()
    plt.figure()
    plt.axis('equal')
    _cs = plt.contour(X_1, Y_1, Z_1)
    plt.grid(True)
    plt.clabel(_cs)
    _x1s = [x[0].item() for x in xs]
    _x2s = [x[1].item() for x in xs]
    plt.plot(_x1s, _x2s, 'r--.', ms=0.5, alpha=0.5)
    plt.axis('equal')
    None
    return


@app.cell
def _(torch):
    x_star = torch.tensor([-1.3636, -0.7273])
    return (x_star,)


@app.cell
def _(f_alt, torch, x_star):
    lrs = torch.linspace(0.06, 0.1, 100)
    es = []
    for _lr in lrs:
        x_6 = torch.tensor([4.0, 4.0], requires_grad=True)
        xs_1 = [x_6.detach().clone()]
        _n = 100
        _optimizer = torch.optim.SGD(params=[x_6], lr=_lr)
        for _i in range(_n):
            _y = f_alt(x_6)
            _y.backward()
            _optimizer.step()
            xs_1.append(x_6.detach().clone())
            _optimizer.zero_grad()
        e = (xs_1[-1] - x_star).norm()
        es.append(e)
    return es, lrs


@app.cell
def _(es, lrs, plt):
    plt.plot(lrs, es)
    plt.xlabel("learning rate")
    plt.ylabel("error")
    return


@app.cell
def _(X_1, Y_1, Z_1, f_alt, plt, torch):
    x_7 = torch.tensor([4.0, 4.0], requires_grad=True)
    xs_2 = [x_7.detach().clone()]
    _lr = 0.065 / 5
    _n = 100
    _optimizer = torch.optim.SGD(params=[x_7], lr=_lr)
    for _i in range(_n):
        _y = f_alt(x_7)
        _y.backward()
        _optimizer.step()
        xs_2.append(x_7.detach().clone())
        _optimizer.zero_grad()
    plt.figure()
    plt.axis('equal')
    _cs = plt.contour(X_1, Y_1, Z_1)
    plt.grid(True)
    plt.clabel(_cs)
    _x1s = [x[0].item() for x in xs_2]
    _x2s = [x[1].item() for x in xs_2]
    plt.plot(_x1s, _x2s, 'r--.', ms=0.5, alpha=0.5)
    plt.axis('equal')
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Fonction de Rosenbrock
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        f(x, y)  = (1-x)^2  + 100 (y - x^2)^2
        $$

        Le minimum est en 

        $$
        (x^*, y^*) = (1.0, 1.0)
        $$

        et 

        $$
        f(x^*, y^*) = 0.0
        $$
        """
    )
    return


@app.function
def banana(xy):
    x = xy[..., 0]
    _y = xy[..., 1]
    a = 1 - x
    b = _y - x * x
    return a * a + 100.0 * b * b


@app.cell
def _(plt, torch):
    x_8 = torch.linspace(-1.0, 3.0, 100)
    _y = torch.linspace(-1.0, 3.0, 100)
    X_2, Y_2 = torch.meshgrid(x_8, _y, indexing='xy')
    _XY = torch.stack((X_2, Y_2), dim=2)
    print(_XY.shape)
    Z_2 = banana(_XY)
    _cs = plt.contour(X_2, Y_2, Z_2, levels=[100, 1000, 2000])
    plt.clabel(_cs)
    plt.grid()
    return


@app.cell
def _(plt, torch):
    x_9 = torch.linspace(0.9, 1.1, 1000)
    _y = torch.linspace(0.9, 1.1, 1000)
    X_3, Y_3 = torch.meshgrid(x_9, _y, indexing='xy')
    _XY = torch.stack((X_3, Y_3), dim=2)
    print(_XY.shape)
    Z_3 = banana(_XY)
    _cs = plt.contour(X_3, Y_3, Z_3, levels=[0.001, 0.1, 1, 5, 10])
    plt.clabel(_cs)
    plt.grid()
    return


@app.cell
def _(plt, torch):
    x_10 = torch.tensor([-1.0, 2.0], requires_grad=True)
    xs_3 = [x_10.detach().clone()]
    _lr = 0.0016
    _n = 100000
    _optimizer = torch.optim.SGD(params=[x_10], lr=_lr)
    for _i in range(_n):
        _y = banana(x_10)
        _y.backward()
        _optimizer.step()
        xs_3.append(x_10.detach().clone())
        _optimizer.zero_grad()
    plt.figure()
    plt.axis('equal')
    x_10 = torch.linspace(-2.0, 4.0, 100)
    _y = torch.linspace(-1.0, 3.0, 100)
    X_4, Y_4 = torch.meshgrid(x_10, _y, indexing='xy')
    _XY = torch.stack((X_4, Y_4), dim=2)
    print(_XY.shape)
    Z_4 = banana(_XY)
    _cs = plt.contour(X_4, Y_4, Z_4, levels=torch.linspace(10, 500, 10))
    plt.grid(True)
    plt.clabel(_cs)
    _x1s = [x[0].item() for x in xs_3]
    _x2s = [x[1].item() for x in xs_3]
    plt.plot(_x1s, _x2s, 'r', alpha=0.1)
    plt.plot(_x1s, _x2s, 'ro', ms=1.0)
    plt.axis('equal')
    None
    return (xs_3,)


@app.cell
def _(xs_3):
    xs_3[-1]
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
