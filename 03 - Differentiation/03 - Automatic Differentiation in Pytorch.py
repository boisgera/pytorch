import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Automatic Differentiation in Pytorch

    [Sébastien Boisgérault], [ITN], Mines Paris - PSL University

    [ITN]: https://itn.dev
    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(torch):
    _x = torch.tensor(0.0, requires_grad=True)
    _sin_x = torch.sin(_x)
    _sin_x.backward()
    _grad_sin_x = _x.grad
    _grad_sin_x
    return


@app.cell
def _():
    return


@app.cell
def _(torch):
    _x = torch.tensor(torch.pi/2, requires_grad=True)
    _y = torch.sin(_x)
    _y.backward()
    _sin_1_x = _x.grad
    _sin_1_x
    return


@app.cell
def _(torch):
    _x = torch.tensor(0.0, requires_grad=True)
    _y = torch.sin(_x) * torch.sin(_x) + torch.cos(_x) * torch.cos(_x)
    _y.backward()
    _x.grad
    return


@app.cell
def _(torch):
    _x_0 = torch.tensor(0.0, requires_grad=True)
    _x_1 = torch.tensor(1.0, requires_grad=True)
    _y = _x_0 + 2.0 * _x_1
    _y.backward()
    _x_0.grad, _x_1.grad
    return


@app.cell
def _(torch):
    _x_0 = torch.tensor(0.0, requires_grad=True)
    _x_1 = torch.tensor(1.0, requires_grad=True)
    _y = _x_0 + 2.0 * _x_1
    _y.backward()

    # And again
    _y = _x_0 + 2.0 * _x_1
    _y.backward()
    _x_0.grad, _x_1.grad # 🤯
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// warning 
    By default, Pytorch accumulates gradients: the newly computed `grad` attributes are added to the stored value. 

    Zero out these attributes between computations if this is not the behavior that you want!

    ///
    """
    )
    return


@app.cell
def _(torch):
    _x_0 = torch.tensor(0.0, requires_grad=True)
    _x_1 = torch.tensor(1.0, requires_grad=True)
    _y = _x_0 + 2.0 * _x_1
    _y.backward()

    # And again
    _x_0.grad = torch.tensor(0.0) # or _x_0.grad.zero_() 
    _x_1.grad = None # also works
    _y = _x_0 + 2.0 * _x_1
    _y.backward()
    _x_0.grad, _x_1.grad # 😊
    return


@app.cell
def _(plt, torch):
    n = 10
    x = torch.rand(n)
    y = 2.0 * x + 0.1 * torch.rand(n)
    plt.figure(figsize=(8, 4.5))
    plt.scatter(x, y, label="data")
    plt.legend()
    plt.grid(True)
    plt.gcf()
    return x, y


@app.cell
def _(plt, torch, x, y):
    a = torch.tensor(1.5)
    y_pred = a * x

    plt.figure(figsize=(8, 4.5))
    plt.scatter(x, y, label="data")
    plt.scatter(x, y_pred, label="predicted")
    plt.legend()
    plt.grid(True)
    plt.gcf()
    return (a,)


@app.cell
def _(a, x, y):
    a.requires_grad=True
    y_pred_ = a * x
    loss = (y - y_pred_) @ (y - y_pred_)
    loss.backward()
    grad = a.grad
    return (loss,)


@app.cell(hide_code=True)
def _(a, loss, mo):
    mo.md(
        rf"""
    Loss function: $a \in \mathbb{{R}} \mapsto \ell(a) \in \mathbb{{R}}$.

    $$
    \ell({a}) = {loss}
    $$

    $$
    \ell'({a}) = {a.grad}
    $$
    """
    )
    return


if __name__ == "__main__":
    app.run()
