import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MPG Model (PyTorch)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | Learning Objectives
    - [ ] Use PyTorch SGD to (try to) solve the least-square iteratively,
    - [ ] Investigate the situation theoretically. Why is it so slow?
    - [ ] Try to improve the convergence with momentum,
    - [ ] Try to improve the convergence with Adam,
    - [ ] Try to improve the convergence with preconditionning.
    ///
    """
    )
    return


@app.cell
def _():
    import math
    return (math,)


@app.cell
def _():
    import torch
    import torch.linalg
    import torch.nn
    return (torch,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns; sns.set_theme()
    return pd, plt


@app.cell
def _(pd):
    df = pd.read_parquet("data/auto_mpg.parquet")
    df
    return (df,)


@app.cell
def _(df, torch):
    weight = torch.tensor(df["weight"]).squeeze()
    return (weight,)


@app.cell
def _(df, torch):
    mpg = torch.tensor(df["mpg"]).squeeze()
    return (mpg,)


@app.cell
def _(mpg, plt, weight):
    plt.scatter(weight, mpg)
    plt.xlabel("weight")
    plt.ylabel("mpg")
    return


@app.cell
def _(weight):
    inv_weight = 1.0 / weight
    return (inv_weight,)


@app.cell
def _(inv_weight, mpg, plt):
    plt.scatter(inv_weight, mpg)
    plt.xlabel("inverse of weight")
    plt.ylabel("mpg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Iterative  Solution of the Linear Model with PyTorch""")
    return


@app.cell
def _(inv_weight, ones, torch):
    A = torch.stack((inv_weight, ones), dim=1)
    return (A,)


@app.cell
def _(A, mpg, torch, weight):
    def _(n=10_000, lr=1e-3):
        alpha_beta = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
        ones = torch.ones_like(weight)
    

        def loss(alpha):
            mpg_pred = A @ alpha_beta
            error = mpg_pred - mpg
            return (error**2).mean()

        optimizer = torch.optim.SGD(params=[alpha_beta], lr=lr)
        for i in range(n):
            l = loss(alpha_beta)
            if (i % int(n / 10)) == 0:
                mpg_pred = A @ alpha_beta
                mean_error = l.sqrt().item()
                relative_error = mean_error / (mpg**2).mean().sqrt().item()
                alpha, beta = alpha_beta
                print(f"i: {i} alpha: {alpha.item():.3g} beta: {beta.item():.3g} error: {relative_error * 100:.1f}%")
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        return alpha.item(), beta.item()


    alpha_1, beta_1 = _()
    return alpha_1, beta_1


@app.cell
def _(alpha_1, beta_1, mpg, plt, weight):
    plt.scatter(weight, mpg)
    plt.xlabel("weight")
    plt.ylabel("mpg")
    plt.scatter(weight, alpha_1 / weight + beta_1)
    return


@app.cell
def _(A, weight):
    S = A.T @ A / len(weight)
    S
    return (S,)


@app.cell
def _(S, torch):
    eigv, _ = torch.linalg.eigh(S)
    lambda_0, lambda_1 = eigv[0].item(), eigv[1].item()
    return lambda_0, lambda_1


@app.cell
def _(lambda_1):
    1.0 / (2.0 * lambda_1) # Good Learning Rate
    return


@app.cell
def _(lambda_0, lambda_1):
    1.0 / (lambda_0 + lambda_1) # Optimal Learning Rate
    return


@app.cell
def _(S, torch):
    cond = torch.linalg.cond(S).item()
    cond
    return (cond,)


@app.cell(hide_code=True)
def _(cond, math, mo):
    mo.md(
        rf"""
    With the good learning rate, in the worst-case, an improvement of the initial error by 50% is achieved when the number of iterations $n$ satisfies


    $$
    \left(1 - \frac{{1}}{{\kappa}}\right)^n \leq \frac{{1}}{{2}}
    $$

    which is equivalent to

    $$
    n \geq -\frac{{\log 2}}{{\log\left(1 - \frac{{1}}{{\kappa}}\right)}} 
    \approx \kappa \log 2
    \approx {int(math.log(2.0) * cond)}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Speed up with Momentum""")
    return


@app.cell
def _(A, mpg, torch, weight):
    def _(n=100_000, lr=0.5):
        alpha_beta = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
        ones = torch.ones_like(weight)
    

        def loss(alpha):
            mpg_pred = A @ alpha_beta
            error = mpg_pred - mpg
            return (error**2).mean()

        optimizer = torch.optim.SGD(params=[alpha_beta], lr=lr, momentum=0.9999)
        for i in range(n):
            l = loss(alpha_beta)
            if (i % int(n / 10)) == 0:
                mpg_pred = A @ alpha_beta
                mean_error = l.sqrt().item()
                relative_error = mean_error / (mpg**2).mean().sqrt().item()
                alpha, beta = alpha_beta
                print(f"i: {i} alpha: {alpha.item():.3g} beta: {beta.item():.3g} error: {relative_error * 100:.1f}%")
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        return alpha.item(), beta.item()


    alpha_2, beta_2 = _()
    return alpha_2, beta_2


@app.cell
def _(alpha_2, beta_2, mpg, plt, weight):
    plt.scatter(weight, mpg)
    plt.xlabel("weight")
    plt.ylabel("mpg")
    plt.scatter(weight, alpha_2 / weight + beta_2)
    return


@app.cell
def _(mo):
    mo.md(r"""### What about Adam?""")
    return


@app.cell
def _(A, mpg, torch, weight):
    def _(n=200_000, lr=0.5):
        alpha_beta = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
        ones = torch.ones_like(weight)
    

        def loss(alpha):
            mpg_pred = A @ alpha_beta
            error = mpg_pred - mpg
            return (error**2).mean()

        optimizer = torch.optim.Adam(params=[alpha_beta], lr=lr)
        for i in range(n):
            l = loss(alpha_beta)
            if (i % int(n / 10)) == 0:
                mpg_pred = A @ alpha_beta
                mean_error = l.sqrt().item()
                relative_error = mean_error / (mpg**2).mean().sqrt().item()
                alpha, beta = alpha_beta
                print(f"i: {i} alpha: {alpha.item():.3g} beta: {beta.item():.3g} error: {relative_error * 100:.1f}%")
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        return alpha.item(), beta.item()


    alpha_3, beta_3 = _()
    return alpha_3, beta_3


@app.cell
def _(alpha_3, beta_3, mpg, plt, weight):
    plt.scatter(weight, mpg)
    plt.xlabel("weight")
    plt.ylabel("mpg")
    plt.scatter(weight, alpha_3 / weight + beta_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Precondition the Data

    i.e. improve its condition number.
    """
    )
    return


@app.cell
def _(weight):
    mean_weight = weight.mean().item()
    return (mean_weight,)


@app.cell
def _(inv_weight, mean_weight):
    inv_weight_scaled = inv_weight * mean_weight
    return (inv_weight_scaled,)


@app.cell
def _(inv_weight_scaled, ones, torch):
    A_scaled = torch.stack((inv_weight_scaled, ones), dim=1)
    return (A_scaled,)


@app.cell
def _(A_scaled, weight):
    S_scaled = (A_scaled.T @ A_scaled) / len(weight)
    S_scaled
    return (S_scaled,)


@app.cell
def _(S_scaled, torch):
    torch.linalg.cond(S_scaled).item()
    return


@app.cell(hide_code=True)
def _(S_scaled, math, mo, torch):
    mo.md(
        rf"""
    With the good learning rate, in the worst-case, an improvement of the initial error by 50% is achieved when the number of iterations $n$ satisfies


    $$
    \left(1 - \frac{{1}}{{\kappa}}\right)^n \leq \frac{{1}}{{2}}
    $$

    which is equivalent to

    $$
    n \geq -\frac{{\log 2}}{{\log\left(1 - \frac{{1}}{{\kappa}}\right)}} 
    \approx \kappa \log 2
    \approx {int(math.log(2.0) * torch.linalg.cond(S_scaled).item())}
    $$
    """
    )
    return


@app.cell
def _(S_scaled, torch):
    eigvals, _ = torch.linalg.eigh(S_scaled)
    eigvals
    return


@app.cell
def _(A_scaled, mean_weight, mpg, torch, weight):
    def _(n=1_000, lr=1/5):
        alpha_beta_scaled = torch.tensor(
            [0.0, 0.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        ones = torch.ones_like(weight)

        def loss(alpha):
            mpg_pred = A_scaled @ alpha_beta_scaled
            error = mpg_pred - mpg
            return (error**2).mean()

        optimizer = torch.optim.SGD(params=[alpha_beta_scaled], lr=lr)
        for i in range(n):
            l = loss(alpha_beta_scaled)
            if (i % int(n / 10)) == 0:
                mpg_pred = A_scaled @ alpha_beta_scaled
                mean_error = l.sqrt().item()
                relative_error = mean_error / (mpg**2).mean().sqrt().item()
                alpha_scaled, beta = alpha_beta_scaled
                print(
                    f"i: {i} alpha: {alpha_scaled.item() * mean_weight:.3g} beta: {beta.item():.3g} error: {relative_error * 100:.1f}%"
                )
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        return alpha_scaled.item() * mean_weight, beta.item()


    alpha_4, beta_4 = _()
    return alpha_4, beta_4


@app.cell
def _(alpha_4, beta_4, mpg, plt, weight):
    plt.scatter(weight, mpg)
    plt.xlabel("weight")
    plt.ylabel("mpg")
    plt.scatter(weight, alpha_4 / weight + beta_4)
    return


@app.cell
def _(mo):
    mo.md(r"""## Neural Network Model""")
    return


@app.cell
def _(mean_weight, mpg, torch, weight):
    weight_ = weight.reshape(-1, 1)
    scaled_weight_ = weight_ / mean_weight
    mean_mpg = mpg.mean().item()
    mpg_ = mpg.reshape(-1, 1)
    scaled_mpg_ = mpg_ / mean_mpg


    def _(n=1_000_000, lr=100.0):
        m, n_ = 2, 2
        model = torch.nn.Sequential(
            torch.nn.Linear(1, m, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(m, n_, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(n_, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            betas=[0.9, 0.999],
        )

        for i in range(n):
            loss = ((model(scaled_weight_) - scaled_mpg_) ** 2).mean()
            if (i % int(n / 10)) == 0:
                relative_error = loss.sqrt().item() / (scaled_mpg_**2).mean().sqrt().item()
                print(f"i: {i} error: {relative_error * 100:.1f}%")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return model


    model = _()
    return mean_mpg, model, scaled_weight_


@app.cell
def _(model):
    dict(model.state_dict())
    return


@app.cell
def _(mean_mpg, mean_weight, model, mpg, plt, scaled_weight_, weight):
    model.eval()
    plt.scatter(weight / mean_weight, mpg / mean_mpg)
    plt.xlabel("weight")
    plt.ylabel("mpg")
    plt.scatter(weight / mean_weight, model(scaled_weight_).detach().numpy())
    return


if __name__ == "__main__":
    app.run()
