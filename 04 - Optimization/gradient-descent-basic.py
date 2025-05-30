import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt
    return plt, torch


@app.function
def f(x, y):
    return 3.0 * _x * _x + 4.0 * _y * _y + _x * _y - _x + 2.0 * _y + 1.0


@app.cell
def _(plt, torch):
    def f_1(x, y):
        return 3.0 * _x * _x + 4.0 * _y * _y + _x * _y - _x + 2.0 * _y + 1.0
    _x = torch.linspace(-1.0, 1.0, 100)
    _y = torch.linspace(-1.0, 1.0, 100)
    X, Y = torch.meshgrid(_x, _y, indexing='xy')
    _Z = f_1(X, Y)
    _cs = plt.contour(X, Y, _Z)
    plt.gca().clabel(_cs, inline=True, fontsize=10)
    plt.colorbar()
    plt.grid(True)
    return X, Y, f_1


@app.cell
def _(f_1):
    def grad_f(x, y):
        _z = f_1(_x, _y)
        _z.backward()
        return [_x.grad.item(), _y.grad.item()]
    return (grad_f,)


@app.cell
def _(X, Y, f_1, grad_f, plt, torch):
    _Z = f_1(X, Y)
    _cs = plt.contour(X, Y, _Z)
    plt.gca().clabel(_cs, inline=True, fontsize=10)
    plt.colorbar()
    plt.grid(True)
    _x = torch.tensor(1.0, requires_grad=True)
    _y = torch.tensor(1.0, requires_grad=True)
    _lr = 0.11884652534268048
    for _i in range(5):
        gx, gy = grad_f(_x, _y)
        _x = torch.tensor(_x.item() - _lr * gx, requires_grad=True)
        _y = torch.tensor(_y.item() - _lr * gy, requires_grad=True)
        plt.plot(_x.item(), _y.item(), 'r.', ms=1.0, alpha=1.0)
    return


@app.cell
def _(X, Y, f_1, plt, torch):
    _Z = f_1(X, Y)
    _cs = plt.contour(X, Y, _Z)
    plt.gca().clabel(_cs, inline=True, fontsize=10)
    plt.colorbar()
    plt.grid(True)
    _x = torch.tensor(1.0, requires_grad=True)
    _y = torch.tensor(1.0, requires_grad=True)
    _lr = 0.001
    _optimizer = torch.optim.SGD(params=[_x, _y], lr=_lr)
    for _i in range(1000):
        _z = f_1(_x, _y)
        _z.backward()
        _optimizer.step()
        plt.plot(_x.item(), _y.item(), 'r.', ms=10.0)
        _optimizer.zero_grad()
    return


@app.cell
def _(X, Y, f_1, plt, torch):
    _Z = f_1(X, Y)
    _cs = plt.contour(X, Y, _Z)
    plt.gca().clabel(_cs, inline=True, fontsize=10)
    plt.colorbar()
    plt.grid(True)
    _x = torch.tensor(1.0, requires_grad=True)
    _y = torch.tensor(1.0, requires_grad=True)
    _lr = 0.237
    _optimizer = torch.optim.SGD(params=[_x, _y], lr=_lr)
    for _i in range(100):
        _z = f_1(_x, _y)
        _z.backward()
        _optimizer.step()
        plt.plot(_x.item(), _y.item(), 'r.', ms=1.0)
        print(_x.item(), _y.item())
        _optimizer.zero_grad()
    return


@app.cell
def _(torch):
    A = torch.tensor([
        [3.0, 0.5], 
        [0.5, 4.0]
    ])
    return (A,)


@app.cell
def _(A, torch):
    torch.linalg.eigh(A)
    return


@app.cell
def _(A, torch):
    lambdas, P = torch.linalg.eigh(A)
    return P, lambdas


@app.cell
def _(lambdas, torch):
    D = torch.diag(lambdas)
    D
    return (D,)


@app.cell
def _(D, P):
    P @ D @ P.T
    return


@app.cell
def _(lambdas):
    lambda_1, lambda_2 = lambdas
    return lambda_1, lambda_2


@app.cell
def _(lambda_2):
    _lr = 0.5 / lambda_2.item()
    _lr
    return


@app.cell
def _(lambda_2):
    lr_lim = 1.0 / lambda_2.item()
    lr_lim
    return


@app.cell
def _(lambda_1, lambda_2):
    best_lr = 1.0 / (lambda_1 + lambda_2)
    best_lr
    return


@app.cell
def _(lambda_1, lambda_2):
    lambda_2 / lambda_1
    return


@app.cell
def _(A, torch):
    k = torch.linalg.cond(A)
    k
    return (k,)


@app.cell
def _(k, torch):
    n_dix = 1.0 / torch.log10(k / (k-1))
    n_dix
    return


if __name__ == "__main__":
    app.run()
