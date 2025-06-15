import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Computation Graph""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from torchviz import make_dot
    return make_dot, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Nota :** we also need `pixi global install graphviz` to have `dot` available.""")
    return


@app.cell
def _(make_dot, nn, torch):
    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))

    x = torch.randn(1, 8)
    y = model(x)

    make_dot(y.mean(), params=dict(model.named_parameters()))
    return


@app.cell
def _(torch):
    x_1 = torch.tensor(1.0, requires_grad=True)
    y_1 = torch.tensor(2.0, requires_grad=True)
    w = x_1 + 2.0 * y_1 + 1.0
    z = w * w + torch.sin(x_1)
    return x_1, y_1, z


@app.cell
def _(make_dot, x_1, y_1, z):
    make_dot(z, params={'x': x_1, 'y': y_1})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Interesting! Probably not exactly what I would like, since the operations that are listed are the gradient computations (and not the forward value computation), but interesting nonetheless.""")
    return


@app.cell
def _(make_dot, torch):
    x_2 = torch.tensor(1.0, requires_grad=True)
    y_2 = torch.sin(x_2)
    make_dot(y_2, params={'x': x_2})
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
