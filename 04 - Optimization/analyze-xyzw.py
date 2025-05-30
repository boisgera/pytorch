import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Prédiction Linéaire de w en fonction de x, y et z
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""


        Obtenir les tenseurs de données
        -------------------------------

        "A la main" tout d'abord:
        """
    )
    return


@app.cell
def _():
    file = open('xyzw.csv', mode='rt', encoding='ascii')
    content = file.read()
    file.close()
    xs = []
    ys = []
    zs = []
    ws = []
    for _i, line in enumerate(content.splitlines()):
        if _i != 0:
            index, x, y, z, w = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))
            ws.append(float(w))
    return ws, xs, ys, zs


@app.cell
def _(ws, xs, ys, zs):
    import torch
    xyz = torch.tensor([xs, ys, zs]).T
    w_1 = torch.tensor(ws)
    (xyz.shape, w_1.shape)
    return torch, w_1


@app.cell
def _(w_1):
    w_1.dtype
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ... ou avec pandas
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    df = pd.read_csv("xyzw.csv")
    del df["Unnamed: 0"]
    df.head(5)
    return df, pd


@app.cell
def _(df, torch):
    xyz_1 = df[['x', 'y', 'z']]
    xyz_1 = torch.tensor(xyz_1.values)
    w_2 = df['w']
    w_2 = torch.tensor(w_2.values)
    (xyz_1.shape, w_2.shape)
    return w_2, xyz_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Recherche de p = [alpha, beta, gamma] optimal
        """
    )
    return


@app.cell
def _(xyz_1):
    xyz_1.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        e = \frac{1}{n} \sum_{i=1}^n (w_i - \hat{w}_i)^2
        $$
        """
    )
    return


@app.cell
def _():
    def error_sum(w, w_pred): # (carré de l') erreur quadratique moyenne
        return ((w-w_pred)**2).sum()

    def error_mean(w, w_pred):
        return ((w-w_pred)**2).mean()
    
    def RMS(w, w_pred):
        return ((w-w_pred)**2).mean().sqrt()
    return RMS, error_mean


@app.cell
def _(RMS, error_mean, torch, w_2, xyz_1):
    lr = 0.155
    p = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD(params=[p], lr=lr)
    for _i in range(100):
        w_pred = xyz_1 @ p
        loss = error_mean(w=w_2, w_pred=w_pred)
        loss.backward()
        print(_i, RMS(w_2, w_pred).item())
        optimizer.step()
        optimizer.zero_grad()
    return (p,)


@app.cell
def _(p):
    p # [alpha, beta, gamma]
    return


@app.cell
def _(p, pd, w_2, xyz_1):
    df_1 = pd.DataFrame()
    df_1['w'] = w_2
    df_1['w (pred)'] = xyz_1 @ p.detach()
    return (df_1,)


@app.cell
def _(torch):
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    t
    return (t,)


@app.cell
def _(t):
    t.detach()
    return


@app.cell
def _(t):
    t.detach()[0] = 0.0
    return


@app.cell
def _(t):
    t.detach().clone()
    return


@app.cell
def _(t):
    t.detach().clone()[0] = 12.0
    t
    return


@app.cell
def _(t):
    t
    return


@app.cell
def _(df_1):
    df_1
    return


@app.cell
def _(df_1):
    import seaborn as sns
    sns.pairplot(df_1)
    None
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
