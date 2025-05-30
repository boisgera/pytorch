import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import torch
    import pandas as pd
    return pd, torch


@app.cell
def _(torch):
    n = 100
    xyz = torch.normal(mean=torch.tensor([1.0, -1.0, 2.0]).expand((n,3)))
    return n, xyz


@app.cell
def _(xyz):
    xyz
    return


@app.cell
def _(torch):
    alpha = [
        torch.tensor(1.0).exp(), 
        torch.pi, 
        0.5 * (1.0 +torch.sqrt(torch.tensor(5.0)))
    ]
    alpha = torch.tensor(alpha)
    alpha
    return (alpha,)


@app.cell
def _(alpha, n, torch, xyz):
    e = torch.normal(mean=torch.zeros(n), std=0.01 * torch.ones(n))
    w = xyz @ alpha + e
    return (w,)


@app.cell
def _(pd, w, xyz):
    df = pd.DataFrame()
    df["x"] = xyz[:, 0]
    df["y"] = xyz[:, 1]
    df["z"] = xyz[:, 2]
    df["w"] = w
    df
    return (df,)


@app.cell
def _(df):
    df.to_csv("xyzw.csv")
    return


@app.cell
def _(df):
    import seaborn as sns
    sns.set_theme()
    sns.pairplot(df)
    return (sns,)


@app.cell
def _(alpha, df, w, xyz):
    df["e"] = w - xyz @alpha
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, sns):
    sns.pairplot(df)
    return


if __name__ == "__main__":
    app.run()
