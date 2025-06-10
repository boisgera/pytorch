import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MPG Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | TODO
    - [ ] Linear and affine prediction from weights only: do it with `torch.linalg.lstsq`, 
    - [ ] scatter the results (`pairplot` or `scatter`)
    - [ ] compute RMS error and associated relative error (as a %)
    - [ ] Replace weight with its inverse (affine model), do it again.
    - [ ] Use pytorch and SGD to get the same result ; find the "best" lr. This is a failure, investigate (theoretically)
    - [ ] Improve the convergence with momentum
    - [ ] Improve the convergence with rescaling (predict how well the algo is gonna do, then do it)
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | TODO
    - [ ] Try power laws with weights?
    - [ ] Improve with extra variables (ex: years ?)
    - [ ] Try other, non-linear architectures (ex: nn) ; take care of over-learning!!!
    ///
    """
    )
    return


@app.cell
def _():
    import torch
    import torch.linalg
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Single Predictor""")
    return


@app.cell
def _(df, torch):
    weight = torch.tensor(df["weight"])
    weight.shape
    return (weight,)


@app.cell
def _(df, torch):
    mpg = torch.tensor(df["mpg"])
    return (mpg,)


@app.cell
def _(mpg, plt, weight):
    plt.scatter(weight, mpg)
    plt.xlabel("weights")
    plt.ylabel("mpg")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Search for $\alpha \in \mathbb{R}$ that provides the best approximation

    $$
    \text{mpg} \approx \alpha \times \text{weight}
    $$

    Understood as, find the minimizer $\alpha$ of the loss function:

    $$
    \text{loss}(\alpha) =  \sum_k (\text{mpg}[k] - \alpha \times \text{weight}[k])^2.
    $$
    """
    )
    return


@app.cell
def _(mpg, torch, weight):
    result = torch.linalg.lstsq(weight.reshape(-1, 1), mpg)
    result
    return (result,)


@app.cell
def _(result):
    alpha_0 = result.solution.item()
    alpha_0
    return


@app.cell
def _():
    return


@app.cell
def _(alpha, mpg, plt, weight):
    plt.scatter(weight, mpg, label="data")
    plt.scatter(weight, alpha*weight, label="predicted")
    plt.xlabel("weights")
    plt.ylabel("mpg")
    plt.legend()
    return


@app.cell(hide_code=True)
def _(alpha, mpg, pd, weight):
    def error_table(data, predicted):
        quadratic_error = ((mpg - alpha * weight) ** 2).mean().sqrt().item()
        relative_error = quadratic_error / (mpg**2).mean().sqrt().item()
        return pd.DataFrame(
            [
                {"name": "quadratic error", "value": f"{quadratic_error:.1f}"},
                {"name": "relative error", "value": f"{100 * relative_error:.1f} %" },
            ]
        )
    return (error_table,)


@app.cell
def _(alpha, error_table, mpg):
    error_table(mpg, alpha * mpg)
    return


@app.cell
def _(mean_error, mo, mpg):
    _relative_error = mean_error / (mpg ** 2).mean().sqrt().item()
    _relative_error
    mo.md(f"Relative error: {_relative_error * 100:.1f}%")
    return


@app.cell
def _(torch, weight):
    offset = torch.ones_like(weight)
    return (offset,)


@app.cell
def _(mpg, offset, torch, weight):
    results = torch.linalg.lstsq(torch.stack((weight, offset), 1), mpg)
    results
    return (results,)


@app.cell
def _(mpg, plt, results, weight):
    alpha_2 = results.solution
    plt.scatter(weight, mpg, label='data')
    plt.scatter(weight, alpha_2[0] * weight + alpha_2[1], label='predicted')
    plt.xlabel('weights')
    plt.ylabel('mpg')
    plt.legend()
    None
    return (alpha_2,)


@app.cell
def _(alpha_2, mpg, weight):
    mpg_pred = alpha_2[0] * weight + alpha_2[1]
    mean_error_1 = ((mpg - mpg_pred) ** 2).mean().sqrt().item()
    _relative_error = mean_error_1 / (mpg ** 2).mean().sqrt().item()
    _relative_error
    print(f'Relative error: {_relative_error * 100:.1f}%')
    return (mpg_pred,)


@app.cell
def _(mpg, mpg_pred, pd):
    df_2 = pd.DataFrame()
    df_2['mpg'] = mpg
    df_2['mpg_pred'] = mpg_pred
    df_2
    return (df_2,)


@app.cell
def _(df_2, sns_1):
    sns_1.pairplot(df_2)
    return


@app.cell
def _(mpg, offset, torch, weight):
    inv_weight = 1.0 / weight
    A = torch.stack((inv_weight, offset), dim=1)
    result_1 = torch.linalg.lstsq(A, mpg)
    result_1
    return inv_weight, result_1


@app.cell
def _(inv_weight, result_1):
    alpha_3 = result_1.solution
    mpg_pred_1 = alpha_3[0] * inv_weight + alpha_3[1]
    return (mpg_pred_1,)


@app.cell
def _(mpg, mpg_pred_1, plt, weight):
    plt.scatter(weight, mpg, label='data')
    plt.scatter(weight, mpg_pred_1, label='predicted')
    plt.xlabel('weights')
    plt.ylabel('mpg')
    plt.legend()
    None
    return


@app.cell
def _(mpg, mpg_pred_1):
    mean_error_2 = ((mpg - mpg_pred_1) ** 2).mean().sqrt().item()
    print(mean_error_2)
    _relative_error = mean_error_2 / (mpg ** 2).mean().sqrt().item()
    _relative_error
    print(f'Relative error: {_relative_error * 100:.1f}%')
    return


@app.cell
def _(mpg, mpg_pred_1, pd, sns_1):
    df_3 = pd.DataFrame()
    df_3['mpg'] = mpg
    df_3['mpg_pred'] = mpg_pred_1
    sns_1.pairplot(df_3)
    return


@app.cell
def _(mpg, offset, torch, weight):
    inv_weight_1 = 1.0 / weight
    A_1 = torch.stack((weight, inv_weight_1, offset), dim=1)
    result_2 = torch.linalg.lstsq(A_1, mpg)
    result_2
    return inv_weight_1, result_2


@app.cell
def _(inv_weight_1, result_2, weight):
    alpha_4 = result_2.solution
    mpg_pred_2 = alpha_4[0] * weight + alpha_4[1] * inv_weight_1 + alpha_4[2]
    return (mpg_pred_2,)


@app.cell
def _(mpg, mpg_pred_2, plt, weight):
    plt.scatter(weight, mpg, label='data')
    plt.scatter(weight, mpg_pred_2, label='predicted')
    plt.xlabel('weights')
    plt.ylabel('mpg')
    plt.legend()
    None
    return


@app.cell
def _(mpg, mpg_pred_2):
    mean_error_3 = ((mpg - mpg_pred_2) ** 2).mean().sqrt().item()
    _relative_error = mean_error_3 / (mpg ** 2).mean().sqrt().item()
    _relative_error
    print(f'Relative error: {_relative_error * 100:.1f}%')
    return


@app.cell
def _(mpg, mpg_pred_2, pd, sns_1):
    df_4 = pd.DataFrame()
    df_4['mpg'] = mpg
    df_4['mpg_pred'] = mpg_pred_2
    sns_1.pairplot(df_4)
    return


@app.cell
def _(mpg, offset, torch, weight):
    clip_weight = torch.clip(weight - 3000, min=0)
    A_2 = torch.stack((weight, clip_weight, offset), dim=1)
    result_3 = torch.linalg.lstsq(A_2, mpg)
    result_3
    return clip_weight, result_3


@app.cell
def _(clip_weight, result_3, weight):
    alpha_5 = result_3.solution
    mpg_pred_3 = alpha_5[0] * weight + alpha_5[1] * clip_weight + alpha_5[2]
    return (mpg_pred_3,)


@app.cell
def _(mpg, mpg_pred_3, plt, weight):
    plt.scatter(weight, mpg, label='data')
    plt.scatter(weight, mpg_pred_3, label='predicted')
    plt.xlabel('weights')
    plt.ylabel('mpg')
    plt.legend()
    None
    return


@app.cell
def _(mpg, mpg_pred_3):
    mean_error_4 = ((mpg - mpg_pred_3) ** 2).mean().sqrt().item()
    _relative_error = mean_error_4 / (mpg ** 2).mean().sqrt().item()
    _relative_error
    print(f'Relative error: {_relative_error * 100:.1f}%')
    return


@app.cell
def _(mpg, mpg_pred_3, offset, pd, torch, weight):
    df_5 = pd.DataFrame()
    df_5['mpg'] = mpg
    df_5['mpg_pred'] = mpg_pred_3
    inv_weight_2 = 1.0 / weight
    A_3 = torch.stack((weight, inv_weight_2, offset), dim=1)
    result_4 = torch.linalg.lstsq(A_3, mpg)
    result_4
    return


@app.cell
def _(mpg, mpg_pred_3):
    mean_error_5 = ((mpg - mpg_pred_3) ** 2).mean().sqrt().item()
    _relative_error = mean_error_5 / (mpg ** 2).mean().sqrt().item()
    _relative_error
    print(f'Relative error: {_relative_error * 100:.1f}%')
    return


@app.cell
def _(mpg, mpg_pred_3, pd, sns_1):
    df_6 = pd.DataFrame()
    df_6['mpg'] = mpg
    df_6['mpg_pred'] = mpg_pred_3
    sns_1.pairplot(df_6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Incremental Solution with PyTorch""")
    return


@app.cell
def _(torch):
    alpha_6 = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    return (alpha_6,)


@app.cell
def _(alpha_6, mpg, offset, torch, weight):
    inv_weight_3 = 1.0 / weight
    A_4 = torch.stack((inv_weight_3, offset), dim=1)

    def loss(alpha):
        mpg_pred = torch.tensordot(A_4, alpha, dims=1)
        delta = mpg - mpg_pred
        return (delta ** 2).mean()
    _lr = 0.5
    _optimizer = torch.optim.SGD(params=[alpha_6], lr=_lr)
    for _i in range(1000):
        _e2 = loss(alpha_6)
        if _i < 100 or _i % 100000 == 0:
            mpg_pred_4 = A_4 @ alpha_6
            mean_error_6 = _e2.sqrt().item()
            _relative_error = mean_error_6 / (mpg ** 2).mean().sqrt().item()
            _relative_error
            print(f'{_i} {alpha_6.detach()} {_relative_error * 100:.1f}%')
        _e2.backward()
        _optimizer.step()
        _optimizer.zero_grad()
    return (A_4,)


@app.cell
def _(alpha_6):
    alpha_6
    return


@app.cell
def _(A_4, weight):
    _n = len(weight)
    print(_n)
    S = A_4.T @ A_4 / _n
    S
    return (S,)


@app.cell
def _(S, torch):
    eigv, _ = torch.linalg.eigh(S)
    eigv
    return (eigv,)


@app.cell
def _(eigv):
    _lr = 1.0 / (2.0 * eigv[1])
    _lr
    return


@app.cell
def _(S, torch):
    cond = torch.linalg.cond(S).item()
    cond
    return (cond,)


@app.cell
def _(cond, torch):
    factor = (1 - 1 / cond)
    print(factor)
    torch.log(torch.tensor(factor))
    return


@app.cell
def _(torch):
    torch.tensor(2.0).log()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \left(1 - \frac{1}{\kappa}\right)^n \leq \frac{1}{2}
    $$
    is equivalent to
    $$
    n \geq -\frac{\log 2}{\log\left(1 - \frac{1}{\kappa}\right)} \approx \kappa \log 2
    $$
    """
    )
    return


@app.cell
def _(cond, torch):
    -torch.tensor(2.0, dtype=torch.float64).log() /  torch.log(torch.tensor(1 - 1/cond, dtype=torch.float64))
    return


@app.cell
def _(cond, torch):
    cond * torch.tensor(2.0).log()
    return


@app.cell
def _(cond, torch):
    cond * torch.tensor(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Try Some momentum""")
    return


@app.cell
def _(torch):
    alpha_7 = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    return (alpha_7,)


@app.cell
def _(alpha_7, mpg, offset, torch, weight):
    inv_weight_4 = 1.0 / weight
    A_5 = torch.stack((inv_weight_4, offset), dim=1)

    def loss_1(alpha):
        mpg_pred = torch.tensordot(A_5, alpha, dims=1)
        delta = mpg - mpg_pred
        return (delta ** 2).mean()
    _lr = 0.5
    momentum = 0.999
    _optimizer = torch.optim.SGD(params=[alpha_7], lr=_lr, momentum=momentum)
    n = 1_000
    for _i in range(n):
        _e2 = loss_1(alpha_7)
        if _i < 100 or _i % n == 0:
            mpg_pred_5 = A_5 @ alpha_7
            mean_error_7 = _e2.sqrt().item()
            _relative_error = mean_error_7 / (mpg ** 2).mean().sqrt().item()
            _relative_error
            print(f'{_i} {alpha_7.detach()} {_relative_error * 100:.1f}%')
        _e2.backward()
        _optimizer.step()
        _optimizer.zero_grad()
    return (inv_weight_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Rescale the data to improve the conditioning""")
    return


@app.cell
def _(df_full):
    df_7 = df_full.copy()
    mpg_1 = df_7['mpg']
    weight_1 = df_7['weight']
    return mpg_1, weight_1


@app.cell
def _(pd):
    df_8 = pd.DataFrame()
    return (df_8,)


@app.cell
def _(df_8, df_full, mpg_1, sns_1, weight_1):
    df_8['1 / weight (rescaled)'] = weight_1.mean() / weight_1
    df_8['origin'] = df_full['origin']
    df_8['mpg (rescaled)'] = mpg_1 / mpg_1.mean()
    sns_1.pairplot(df_8, hue='origin')
    None
    return


@app.cell
def _(df_8, torch):
    A_6 = torch.stack((torch.tensor(df_8['1 / weight (rescaled)'].values), torch.ones(len(df_8), dtype=torch.float64)), dim=1)
    return (A_6,)


@app.cell
def _(A_6, df_8, torch):
    S_1 = A_6.T @ A_6 / len(df_8)
    torch.linalg.cond(S_1).item()
    return (S_1,)


@app.cell
def _(S_1, torch):
    ev, _ = torch.linalg.eigh(S_1)
    ev
    return (ev,)


@app.cell
def _(ev):
    _lr = 0.5 / ev[1].item()
    _lr
    return


@app.cell
def _(torch):
    alpha_scaled = torch.tensor([ 0.0, 0.0 ], dtype=torch.float64, requires_grad=True)
    return (alpha_scaled,)


@app.cell
def _(A_6, alpha_7, alpha_scaled, df_8, torch):
    mpg_rescaled = torch.tensor(df_8['mpg (rescaled)'], dtype=torch.float64)

    def loss_2(alpha_scaled):
        mpg_rescaled_pred = torch.tensordot(A_6, alpha_scaled, dims=1)
        delta = mpg_rescaled - mpg_rescaled_pred
        return (delta ** 2).mean()
    _lr = 0.2
    _optimizer = torch.optim.SGD(params=[alpha_scaled], lr=_lr)
    for _i in range(100000):
        _e2 = loss_2(alpha_scaled)
        if _i < 100 or _i % 100000 == 0:
            mpg_pred_6 = A_6 @ alpha_scaled
            mean_error_8 = _e2.sqrt().item()
            _relative_error = mean_error_8 / (mpg_rescaled ** 2).mean().sqrt().item()
            _relative_error
            print(f'{_i} {alpha_scaled.detach()} {_relative_error * 100:.1f}%')
        _e2.backward()
        _optimizer.step()
        _optimizer.zero_grad()
    print(f'{_i} {alpha_7.detach()} {_relative_error * 100:.1f}%')
    return


@app.cell
def _(alpha_scaled, df_full, torch, weight_1):
    print(alpha_scaled)
    alpha_8 = alpha_scaled.detach().clone() * torch.tensor(df_full['mpg'].mean())
    print(alpha_8)
    alpha_8[0] = alpha_8[0] * weight_1.mean()
    alpha_8
    return (alpha_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Contour""")
    return


@app.cell
def _(A_6, mpg_1, plt, torch):
    def _e2(alphas):
        mpg_pred = torch.einsum('ij,...j->...i', A_6, alphas)
        return ((mpg_pred - mpg_1) ** 2).mean(dim=-1)
    alpha_0 = torch.linspace(0.01, 100000, 1000, dtype=torch.float64)
    alpha_1 = torch.linspace(-5.0, 5.0, 1000, dtype=torch.float64)
    Alpha_0, Alpha_1 = torch.meshgrid(alpha_0, alpha_1, indexing='ij')
    Alpha = torch.stack((Alpha_0, Alpha_1), dim=2)
    plt.figure(figsize=(12.0, 4.0))
    cs = plt.contour(Alpha_0, Alpha_1, _e2(Alpha), levels=[17, 18, 19, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.clabel(cs, inline=1, fontsize=10)
    None
    return


@app.cell
def _(alpha_8, inv_weight_4, torch):
    with torch.no_grad():
        mpg_pred_7 = alpha_8[0].item() * inv_weight_4 + alpha_8[1].item()
    return (mpg_pred_7,)


@app.cell
def _(mpg_1, mpg_pred_7):
    mean_error_9 = ((mpg_1 - mpg_pred_7) ** 2).mean().sqrt().item()
    _relative_error = mean_error_9 / (mpg_1 ** 2).mean().sqrt().item()
    _relative_error
    print(f'Relative error: {_relative_error * 100:.1f}%')
    return


@app.cell
def _(mpg_1, mpg_pred_7, plt, weight_1):
    plt.scatter(weight_1, mpg_1, label='data')
    plt.scatter(weight_1, mpg_pred_7.detach(), label='predicted')
    plt.xlabel('weights')
    plt.ylabel('mpg')
    plt.legend()
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Power Laws""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Match 
    $$
    \mathrm{mpg} = a \mathrm{weight}^{b}
    $$
    directly or using the linear match
    $$
    \log \mathrm{mpg} = \log a + b \log(\mathrm{weight})
    $$

    ```{warning}
    The results will be different since the loss is not the same value in these two approaches.
    ```
    """
    )
    return


@app.cell
def _(df_full):
    df_9 = df_full[['mpg']]
    df_9
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Multiple Predictors""")
    return


@app.cell
def _(df_full):
    keys = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    df_10 = df_full.copy()
    df_10[keys]
    return df_10, keys


@app.cell
def _(df_10, keys, torch):
    X_1 = torch.tensor(df_10[keys].values)
    X_1
    return (X_1,)


@app.cell
def _(df_10, m, torch):
    USA = torch.tensor((df_10['origin'] == 'USA').values).reshape((m, 1))
    Europe = torch.tensor((df_10['origin'] == 'Europe').values).reshape((m, 1))
    Asia = torch.tensor((df_10['origin'] == 'Asia').values).reshape((m, 1))
    X_2 = torch.cat([USA, Europe, Asia], 1)
    X_2
    return (X_2,)


@app.cell
def _(m, torch):
    X_3 = torch.ones((m, 1), dtype=torch.float64)
    X_3
    return (X_3,)


@app.cell
def _(X, X_1, X_2, X_3, torch):
    m, _n = X.shape
    X_4 = torch.cat((X_1, X_2, X_3), 1)
    X_4
    return X_4, m


@app.cell
def _(df_10, torch):
    y_1 = torch.tensor(df_10['mpg'].values)
    y_1
    return (y_1,)


@app.cell
def _(X_4, torch, y_1):
    r_1 = torch.linalg.lstsq(X_4, y_1)
    return (r_1,)


@app.cell
def _(r_1):
    A_7 = r_1.solution
    A_7
    return (A_7,)


@app.cell
def _(A_7, X_4, torch, y_1):
    y_hat = torch.tensordot(X_4, A_7, dims=1)
    e = y_1 - y_hat
    return (y_hat,)


@app.cell
def _(plt, y_1, y_hat):
    plt.hist(y_1, label='MPG value')
    plt.hist(y_hat, label='predicted MPG value')
    plt.legend()
    None
    return


@app.cell
def _(y_1, y_hat):
    e_1 = y_1 - y_hat
    return (e_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    If $\mu$ denotes the mean of $e$ 
    $$
    \mu = \frac{1}{n} \sum_{k=1}^n e_k
    $$
    and $\sigma$ its (biased, uncorrected) standard deviation
    $$
    \sigma = \sqrt{\frac{1}{n}\sum_{k=1}^n (e_k - \mu)^2}
    $$
    then the mean quadratic error $\ell$ (our loss function) satisfies
    $$
    \ell = \sqrt{\frac{1}{n}\sum_{k=1}^n e_k^2} = \sqrt{\mu^2 + \sigma^2}.
    $$
    """
    )
    return


@app.cell
def _(e_1):
    def loss_3(e):
        return (e.mean() ** 2 + e.std(unbiased=False) ** 2).sqrt().item()

    def loss_alt(e):
        return (e ** 2).mean().sqrt().item()
    (loss_3(e_1), loss_alt(e_1))
    return (loss_3,)


@app.cell
def _(e_1, loss_3, y_1):
    print('Mean MPG value:', y_1.mean().item())
    print('Mean MPG prediction error:', e_1.abs().mean().item())
    print('Mean (quadratic) MPG prediction error:', loss_3(e_1))
    return


@app.cell
def _(df_10, y_hat):
    df_10['mpg (pred.)'] = y_hat
    df_10.to_csv('data/auto_mpg_pred.csv')
    return


@app.cell
def _(A_7):
    A_7
    return


@app.cell
def _(A_7, keys):
    for k, slope in zip(keys + ['USA', 'Europe', 'Asia'] + ['bias'], A_7):
        print(k, slope.item())
    return


@app.cell
def _(X_4, plt, y_1):
    plt.scatter(X_4[:, 0], y_1)
    return


@app.cell
def _(df_10, sns_1):
    df_hat = df_10.copy()
    sns_1.pairplot(df_hat, height=2.0, hue='origin')
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **TODO.**

      - Reorder the whole thing? Simple things first (linear model, original model, etc.) THEN assess limitations?
      - Measure the performance (lstsq)
      - Do it for the original variables as well as the inverted ones (should be better, right?)
      - Try nonlinear schemes to deal with some $1/x^{\alpha}$ with unknown alpha stuff.
    """
    )
    return


@app.cell
def _(df_10):
    def f(x):
        return 1.0 / x
    df_10['displacement'] = df_10['displacement'].transform(f)
    df_10['horsepower'] = df_10['horsepower'].transform(f)
    df_10['weight'] = df_10['weight'].transform(f)
    return


if __name__ == "__main__":
    app.run()
