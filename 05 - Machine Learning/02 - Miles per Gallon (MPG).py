import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MPG Prediction""")
    return


@app.cell
def _():
    import pandas as pd
    import torch
    return pd, torch


@app.cell
def _(pd):
    df = pd.read_csv("data.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    def region(i):
        if i == 1:
            return "USA"
        elif i == 2:
            return "Europe"
        elif i == 3:
            return "Asia"
    # df['origin'] = df['origin'].transform(region)
    # df
    return


@app.cell
def _(df):
    df_1 = df.dropna().reset_index(drop=True)
    df_1
    return (df_1,)


@app.cell
def _(df_1):
    import seaborn as sns
    sns.pairplot(df_1, hue='origin')
    return (sns,)


@app.cell
def _(df_1, torch):
    torch.tensor(df_1['mpg'])
    return


@app.cell
def _(df_1):
    df_USA = df_1[df_1['origin'] == 'USA']
    df_EUA = df_1[df_1['origin'] != 'USA']
    return df_EUA, df_USA


@app.cell
def _(df_USA, sns):
    sns.pairplot(df_USA)
    return


@app.cell
def _(df_EUA, sns):
    sns.pairplot(df_EUA)
    return


@app.cell
def _(df_1, torch):
    torch.tensor(df_1[['weight', 'mpg']].values)
    return


@app.cell
def _(df_1, torch):
    mpg = torch.tensor(df_1['mpg'])
    weight = torch.tensor(df_1['weight'])
    mpg.dtype
    return mpg, weight


@app.cell
def _(mpg, torch, weight):
    alpha = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    _lr = 1e-07
    _optimizer = torch.optim.SGD(params=[alpha], lr=_lr)
    _n = 1000
    for _i in range(_n):
        _mean_squared_error = ((alpha * weight - mpg) ** 2).mean()
        _mean_squared_error.backward()
        _optimizer.step()
        if _i % 100 == 0:
            print(alpha.item())
        _optimizer.zero_grad()
    alpha = alpha.detach()
    _mean_relative_error = ((alpha * weight - mpg).abs() / mpg).mean().item()
    print(f'erreur relative moyenne: {100 * _mean_relative_error:.2f} %')
    return (alpha,)


@app.cell
def _(alpha, mpg, pd, weight):
    df_lin = pd.DataFrame()
    df_lin["weight"] = weight
    df_lin["mpg (pred)"] = alpha * weight
    df_lin["mpg"] = mpg
    df_lin
    return


@app.cell
def _(alpha, mpg, weight):
    import matplotlib.pyplot as plt
    plt.plot(weight, mpg, "bo", label="mpg (data)")
    plt.plot(weight, alpha * weight, "ro", label="mpg (pred.)")
    plt.grid(True)
    plt.xlabel("weight")
    plt.ylabel("mpg")
    plt.legend()
    return (plt,)


@app.cell
def _(mpg, torch, weight):
    alpha_1 = torch.tensor(-0.01, requires_grad=True, dtype=torch.float64)
    beta = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    _lr = 1e-07
    _optimizer = torch.optim.SGD(params=[alpha_1, beta], lr=_lr)
    _n = 100000
    for _i in range(_n):
        _mean_squared_error = ((alpha_1 * weight + beta - mpg) ** 2).mean()
        _mean_squared_error.backward()
        _optimizer.step()
        if _i % 10000 == 0:
            print(alpha_1.item(), beta.item())
        _optimizer.zero_grad()
    alpha_1 = alpha_1.detach()
    beta = beta.detach()
    _mean_relative_error = ((alpha_1 * weight + beta - mpg).abs() / mpg).mean().item()
    print(f'erreur relative moyenne: {100 * _mean_relative_error:.2f} %')
    return


@app.cell
def _(torch, weight):
    _M = torch.stack((weight, torch.ones_like(weight))).to(torch.float64)
    A = _M @ _M.T
    A
    return (A,)


@app.cell
def _(A, torch):
    torch.linalg.cond(A)
    return


@app.cell
def _(torch):
    grad = torch.tensor([1.0, -0.9, 1.1, -1.2, 0.99, -0.8, 1.0, -1.0, 0.9, -1.1])
    return (grad,)


@app.cell
def _(grad):
    grad_sum = 0.0
    print(grad_sum)
    for _i, g in enumerate(grad):
        grad_sum = grad_sum + g
        print('gradient:', g, 'gradient lissé & amplifié:', grad_sum)
    return


@app.cell
def _(mpg, torch, weight):
    alpha_2 = torch.tensor(-0.01, requires_grad=True, dtype=torch.float64)
    beta_1 = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    _lr = 1e-07
    momentum = 0.999
    _optimizer = torch.optim.SGD(params=[alpha_2, beta_1], lr=_lr, momentum=momentum)
    _n = 1000000
    for _i in range(_n):
        _mean_squared_error = ((alpha_2 * weight + beta_1 - mpg) ** 2).mean()
        _mean_squared_error.backward()
        _optimizer.step()
        if _i % 10000 == 0:
            print(alpha_2.item(), beta_1.item())
        _optimizer.zero_grad()
    alpha_2 = alpha_2.detach()
    beta_1 = beta_1.detach()
    _mean_relative_error = ((alpha_2 * weight + beta_1 - mpg).abs() / mpg).mean().item()
    print(f'erreur relative moyenne: {100 * _mean_relative_error:.2f} %')
    return alpha_2, beta_1


@app.cell
def _(alpha_2, beta_1, mpg, plt, weight):
    plt.plot(weight, mpg, 'bo', label='mpg (data)')
    plt.plot(weight, alpha_2 * weight + beta_1, 'ro', label='mpg (pred.)')
    plt.grid(True)
    plt.xlabel('weight')
    plt.ylabel('mpg')
    plt.legend()
    return


@app.cell
def _(weight):
    normalized_weight = weight / 3000.0
    return (normalized_weight,)


@app.cell
def _(normalized_weight):
    normalized_weight.mean()
    return


@app.cell
def _(normalized_weight, torch, weight):
    _M = torch.stack((normalized_weight, torch.ones_like(weight)))
    A_1 = _M @ _M.T / len(weight)
    torch.linalg.cond(A_1)
    return (A_1,)


@app.cell
def _(A_1, torch):
    torch.linalg.eigh(A_1)
    return


@app.cell
def _(normalized_weight, torch):
    weight_beyond_3000_normalized = torch.max(normalized_weight - 1.0, torch.zeros_like(normalized_weight))
    return (weight_beyond_3000_normalized,)


@app.cell
def _(mpg, normalized_weight, torch, weight, weight_beyond_3000_normalized):
    alpha_normalized = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    gamma_normalized = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    beta_2 = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    _lr = 0.1
    _optimizer = torch.optim.SGD(params=[alpha_normalized, gamma_normalized, beta_2], lr=_lr)
    _n = 100000
    for _i in range(_n):
        _mean_squared_error = ((alpha_normalized * normalized_weight + gamma_normalized * weight_beyond_3000_normalized + beta_2 - mpg) ** 2).mean()
        _mean_squared_error.backward()
        _optimizer.step()
        if _i % 10000 == 0:
            print(alpha_normalized.item() / 3000.0, gamma_normalized.item() / 3000.0, beta_2.item())
        _optimizer.zero_grad()
    alpha_3 = alpha_normalized.detach() / 3000.0
    gamma = gamma_normalized.detach() / 3000.0
    weight_beyond_3000 = weight_beyond_3000_normalized * 3000.0
    beta_2 = beta_2.detach()
    _mean_relative_error = ((alpha_3 * weight + gamma * weight_beyond_3000 + beta_2 - mpg).abs() / mpg).mean().item()
    print(f'erreur relative moyenne: {100 * _mean_relative_error:.2f} %')
    return alpha_3, alpha_normalized, beta_2, gamma, weight_beyond_3000


@app.cell
def _(alpha_3, beta_2, gamma, mpg, plt, weight, weight_beyond_3000):
    plt.plot(weight, mpg, 'bo', label='mpg (data)')
    plt.plot(weight, alpha_3 * weight + gamma * weight_beyond_3000 + beta_2, 'ro', label='mpg (pred.)')
    plt.grid(True)
    plt.xlabel('weight')
    plt.ylabel('mpg')
    plt.legend()
    return


@app.cell
def _(alpha_normalized, mpg, normalized_weight, torch, weight):
    _lr = 0.1
    _optimizer = torch.optim.SGD(params=[alpha_normalized], lr=_lr)
    _n = 1000
    for _i in range(_n):
        _mean_squared_error = ((alpha_normalized / normalized_weight - mpg) ** 2).mean()
        _mean_squared_error.backward()
        _optimizer.step()
        if _i % 100 == 0:
            print(alpha_normalized.item() * 3000.0)
        _optimizer.zero_grad()
    alpha_4 = alpha_normalized.detach() * 3000.0
    _mean_relative_error = ((alpha_4 / weight - mpg).abs() / mpg).mean().item()
    print(f'erreur relative moyenne: {100 * _mean_relative_error:.2f} %')
    return (alpha_4,)


@app.cell
def _(alpha_4, mpg, plt, weight):
    plt.plot(weight, mpg, 'bo', label='mpg (data)')
    plt.plot(weight, alpha_4 / weight, 'ro', label='mpg (pred.)')
    plt.grid(True)
    plt.xlabel('weight')
    plt.ylabel('mpg')
    plt.legend()
    return


@app.cell
def _(df_1, torch):
    year = torch.tensor(df_1['model_year']).to(torch.float64)
    year_normalized = (year - year.mean()) / year.std()
    year_normalized
    return (year_normalized,)


@app.cell
def _(mpg, normalized_weight, plt, torch, weight, year_normalized):
    alpha_5 = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    beta_3 = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    gamma_1 = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    _lr = 0.1
    _optimizer = torch.optim.SGD(params=[alpha_5, beta_3, gamma_1], lr=_lr)
    _n = 10000
    for _i in range(_n):
        _mean_squared_error = ((alpha_5 / normalized_weight + beta_3 * year_normalized + gamma_1 - mpg) ** 2).mean()
        _mean_squared_error.backward()
        _optimizer.step()
        if _i % 100 == 0:
            print(alpha_5.item(), beta_3.item(), gamma_1.item())
        _optimizer.zero_grad()
    alpha_5 = alpha_5.detach()
    beta_3 = beta_3.detach()
    gamma_1 = gamma_1.detach()
    _mean_relative_error = ((alpha_5 / normalized_weight + beta_3 * year_normalized + gamma_1 - mpg).abs() / mpg).mean().item()
    print(f'erreur relative moyenne: {100 * _mean_relative_error:.2f} %')
    plt.plot(weight, mpg, 'bo', label='mpg (data)')
    plt.plot(weight, alpha_5 / weight, 'ro', label='mpg (pred.)')
    plt.grid(True)
    plt.xlabel('weight')
    plt.ylabel('mpg')
    plt.legend()
    return alpha_5, beta_3, gamma_1


@app.cell
def _(
    alpha_5,
    beta_3,
    gamma_1,
    mpg,
    normalized_weight,
    plt,
    weight,
    year_normalized,
):
    plt.plot(weight, mpg, 'bo', label='mpg (data)')
    plt.plot(weight, alpha_5 / normalized_weight + beta_3 * year_normalized + gamma_1, 'ro', label='mpg (pred.)')
    plt.grid(True)
    plt.xlabel('weight')
    plt.ylabel('mpg')
    plt.legend()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
