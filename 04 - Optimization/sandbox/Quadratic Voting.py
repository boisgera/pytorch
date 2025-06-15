import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Quadratic Voting""")
    return


@app.cell
def _():
    import math

    import torch
    import numpy as np
    import pandas as pd
    return np, pd, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Given a list of sensitivity $\alpha$ to three issues, for example $\alpha = (1.0, 2.0, 3.0)$,
    determine how the voter should split its vote among the three issues
    $$
    x = (x_0, x_1, x_2), \; x_0 + x_1 + x_2 = 1, \; x_0 \geq 0, \; x_1 \geq 0, \; x_2 \geq 0. 
    $$
    to maximize the associated **utility**
    $$
    U(x)
    :=
    \alpha_1 x_1 + \alpha_2 x_2 + \alpha_3 x_3.
    $$
    """
    )
    return


@app.cell
def _(torch):
    alpha = torch.tensor([1.0, 3.0, 2.0])

    def U(x):
        "Utility function (supports batching)"
        return torch.tensordot(x, alpha, dims=1)
    return (U,)


@app.cell
def _(U, torch):
    U(x=torch.tensor([1/3, 1/3, 1/3]))
    return


@app.cell
def _(U, torch):
    U(x=torch.tensor([0.0, 1.0, 0.0]))
    return


@app.cell
def _(U, torch):
    U(torch.tensor([1.0, 0.0, 0.0]))
    return


@app.cell
def _(torch):
    batch = torch.tensor([
        [1/3, 1/3, 1/3],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    return (batch,)


@app.cell
def _(U, batch):
    U(batch)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Unconstrained Optimization""")
    return


@app.cell
def _(U, pd, torch):
    x = torch.tensor([1/3, 1/3, 1/3], requires_grad=True)
    optimizer = torch.optim.SGD(params=[x], lr=0.01, maximize=True)

    df = []
    for i in range(100):
        optimizer.zero_grad()
        utility = U(x)
        x0, x1, x2 = x.detach().numpy()
        df.append({"i": i, "x0": x0, "x1": x1, "x2": x2, "utility": utility.item()})
        utility.backward()
        optimizer.step()

    pd.DataFrame(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Projection Methods : kinda hard""")
    return


@app.cell
def _(U, pd, torch):
    x_1 = torch.tensor([1 / 3, 1 / 3, 1 / 3], requires_grad=True)
    optimizer_1 = torch.optim.SGD(params=[x_1], lr=0.01, maximize=True)
    df_1 = []
    for i_1 in range(100):
        xp = x_1 - x_1.mean() + torch.ones(3) / 3
        optimizer_1.zero_grad()
        utility_1 = U(xp)
        x0_1, x1_1, x2_1 = x_1.detach().numpy()
        df_1.append({'i': i_1, 'x0': x0_1, 'x1': x1_1, 'x2': x2_1, 'utility': utility_1.item()})
        utility_1.backward()
        optimizer_1.step()
    pd.DataFrame(df_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Penalization""")
    return


@app.cell
def _(U, pd, torch):
    x_2 = torch.tensor([1 / 3, 1 / 3, 1 / 3], requires_grad=True)
    mu = 100
    optimizer_2 = torch.optim.SGD(params=[x_2], lr=0.001, maximize=True)
    df_2 = []
    for i_2 in range(1000):
        optimizer_2.zero_grad()
        utility_2 = U(x_2)
        penalized_utility = utility_2 - mu * ((x_2.sum() - 1.0) ** 2 + min(x_2[0], 0) ** 2 + min(x_2[1], 0) ** 2 + min(x_2[2], 0) ** 2)
        x0_2, x1_2, x2_2 = x_2.detach().numpy()
        df_2.append({'i': i_2, 'x0': x0_2, 'x1': x1_2, 'x2': x2_2, 'utility': utility_2.item(), 'penalized_utility': penalized_utility.item()})
        penalized_utility.backward()
        optimizer_2.step()
    pd.DataFrame(df_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Reparametrization""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Use 3 "free" parameters $(y_0, y_1, y_2)$ in $\mathbb{R}^3$, compute $x$ from them with:

    $$
    x_i = \frac{\exp y_i}{\exp y_1 + \exp y_2 + \exp y_3}.
    $$

    By design, $x$ belongs to the allowed parameter sets (⚠️ $x>0$!). 

    Optimize wrt $y$.
    """
    )
    return


@app.cell
def _(U, pd, torch):
    y = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    optimizer_3 = torch.optim.SGD(params=[y], lr=0.1, maximize=True)
    df_3 = []
    for i_3 in range(1000):
        optimizer_3.zero_grad()
        x_3 = y.exp() / y.exp().sum()
        utility_3 = U(x_3)
        x0_3, x1_3, x2_3 = x_3.detach().numpy()
        df_3.append({'i': i_3, 'x0': x0_3, 'x1': x1_3, 'x2': x2_3, 'utility': utility_3.item()})
        utility_3.backward()
        optimizer_3.step()
    pd.DataFrame(df_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Quadratic voting""")
    return


@app.cell
def _(torch):
    alpha_1 = torch.tensor([1.0, 3.0, 2.0])

    def U_1(x):
        """Utility function (supports batching)"""
        return torch.tensordot(x.sqrt(), alpha_1, dims=1)
    return U_1, alpha_1


@app.cell
def _(U_1, pd, torch):
    y_1 = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    optimizer_4 = torch.optim.SGD(params=[y_1], lr=0.1, maximize=True)
    df_4 = []
    for i_4 in range(1000):
        optimizer_4.zero_grad()
        x_4 = y_1.exp() / y_1.exp().sum()
        utility_4 = U_1(x_4)
        x0_4, x1_4, x2_4 = x_4.detach().numpy()
        df_4.append({'i': i_4, 'x0': x0_4, 'x1': x1_4, 'x2': x2_4, 'utility': utility_4.item()})
        utility_4.backward()
        optimizer_4.step()
    pd.DataFrame(df_4)
    return utility_4, x_4


@app.cell
def _(x_4):
    impact = x_4.detach().sqrt()
    impact
    return (impact,)


@app.cell
def _(alpha_1, impact, torch):
    norm = torch.linalg.vector_norm
    impact / norm(impact) * norm(alpha_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Using torch nn modules""")
    return


@app.cell
def _(Softmax, torch):
    class Utility(torch.nn.Module): 
        def __init__(self, alpha):
            super().__init__()
            self.alpha = alpha
            self.logits = torch.nn.parameter.Parameter(torch.tensor([0.0, 0.0, 0.0]))
            self.softmax = Softmax(dim=-1)
        def forward(self):
            return torch.tensordot(self.x, self.alpha, dims=1)
        def get_x(self):
            return self.softmax(self.logits)
        x = property(get_x)
    return (Utility,)


app._unparsable_cell(
    r"""
    utility = Utility(torch.tensor([1.0, 3.0, 2.0]))
    print(f{\"utility.logits)
    print(utility.x)
    utility()
    """,
    name="_"
)


@app.cell
def _(SGD, utility_4):
    optimizer_5 = SGD(utility_4.parameters(), lr=1.0)
    return


@app.cell
def _(SGD, Utility):
    utility_5 = Utility([1.0, 3.0, 2.0])
    optimizer_6 = SGD(utility_5.parameters(), lr=1.0)
    print(utility_5.x.detach().numpy())
    for i_5 in range(10000):
        optimizer_6.zero_grad()
        loss = -utility_5()
        loss.backward()
        optimizer_6.step()
        if i_5 % 1000 == 0:
            print(utility_5.x.detach().numpy())
    return


@app.cell
def _(Parameter, Softmax, rand, sqrt, torch):
    class Utility_1(torch.nn.Module):

        def __init__(self, alpha, quadratic=False):
            super().__init__()
            self.quadratic = quadratic
            self.alpha = alpha
            self.logits = Parameter(rand(3))
            self.softmax = Softmax(dim=-1)

        def forward(self):
            x = self.x
            if self.quadratic:
                x = sqrt(x)
            return x @ self.alpha

        def get_x(self):
            return self.softmax(self.logits)
        x = property(get_x)
    return (Utility_1,)


@app.cell
def _(SGD, Utility_1, np):
    utility_6 = Utility_1([1.0, 3.0, 2.0], quadratic=True)
    optimizer_7 = SGD(utility_6.parameters(), lr=1.0)
    x_5 = utility_6.x.detach().numpy()
    print(x_5, np.sqrt(x_5))
    for i_6 in range(10000):
        optimizer_7.zero_grad()
        loss_1 = -utility_6()
        loss_1.backward()
        optimizer_7.step()
        if i_6 % 1000 == 0:
            x_5 = utility_6.x.detach().numpy()
            s = np.sqrt(x_5)
            print(x_5, s)
    return (s,)


@app.cell
def _(s):
    s[1] / s[0]
    return


@app.cell
def _(s):
    s[2] / s[0]
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
