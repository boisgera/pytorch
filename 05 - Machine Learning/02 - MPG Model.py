import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MPG Linear Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Learning Objectives
    - [ ] Learn what `torch.linalg.lstsq` (Least-Square Problem) is about,
    - [ ] Discover how it solves linear prediction with root mean square error,
    - [ ] Try some linear and affine prediction of mpg based on weights only,
    - [ ] Display the results (`pairplot` or `scatter`)
    - [ ] compute RMS error and associated relative error (as a %)
    - [ ] Replace weight with its inverse (affine model), do it again.
    - [ ] Take into account the production year,
    - [ ] Take into account the origin.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
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
    return pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_parquet("data/auto_mpg.parquet")
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Linear Model wrt Weight
    """)
    return


@app.cell
def _(df, sns):
    sns.pairplot(df[["weight", "mpg"]])
    return


@app.cell
def _(df, torch):
    weight = torch.tensor(df["weight"])
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Search for $\alpha \in \mathbb{R}$ that provides the best approximation

    $$
    \text{mpg} \approx \alpha \times \text{weight}
    $$

    Understood as, find the minimizer $\alpha$ of the loss function:

    $$
    \text{loss}(\alpha) =  \sum_k (\text{mpg}[k] - \alpha \times \text{weight}[k])^2.
    $$
    """)
    return


@app.cell
def _(mpg, torch, weight):
    result_0 = torch.linalg.lstsq(weight.reshape(-1, 1), mpg)
    result_0
    return (result_0,)


@app.cell
def _(result_0):
    alpha_0 = result_0.solution.item()
    alpha_0
    return (alpha_0,)


@app.cell
def _(alpha_0, weight):
    predicted_0 = alpha_0 * weight
    return (predicted_0,)


@app.cell
def _(mpg, plt, predicted_0, weight):
    plt.scatter(weight, mpg, label="data")
    plt.scatter(weight, predicted_0, label="predicted")
    plt.xlabel("weights")
    plt.ylabel("mpg")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_0, weight):
    plt.scatter(weight, predicted_0 - mpg, label="prediction error")
    plt.xlabel("weights")
    plt.ylabel("mpg error")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_0, sns):
    sns.displot(predicted_0 - mpg, stat="probability")
    plt.xlabel("error")
    return


@app.cell(hide_code=True)
def _(pd):
    def error_table(data, predicted):
        quadratic_error = ((data - predicted) ** 2).mean().sqrt().item()
        relative_error = quadratic_error / (data**2).mean().sqrt().item()
        return pd.DataFrame(
            [
                {"name": "quadratic error", "value": f"{quadratic_error:.1f}"},
                {"name": "relative error", "value": f"{100 * relative_error:.1f} %" },
            ]
        )

    return (error_table,)


@app.cell
def _(error_table, mpg, predicted_0):
    error_table(mpg, predicted_0)
    return


@app.cell
def _(mpg, pd, predicted_0, sns):
    sns.pairplot(pd.DataFrame({"mpg": mpg, "predicted": predicted_0}))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affine Model wrt Weight
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Search for $\alpha, \, \beta \in \mathbb{R}$ that provides the best approximation

    $$
    \text{mpg} \approx \alpha \times \text{weight} + \beta
    $$

    Understood as, find the minimizer $\alpha$ of the loss function:

    $$
    \text{loss}(\alpha) =  \sum_k (\text{mpg}[k] - \alpha \times \text{weight}[k] - \beta)^2.
    $$
    """)
    return


@app.cell
def _(torch, weight):
    ones = torch.ones_like(weight)
    return (ones,)


@app.cell
def _(mpg, ones, torch, weight):
    result_1 = torch.linalg.lstsq(torch.stack((weight, ones), 1), mpg)
    result_1
    return (result_1,)


@app.cell
def _(result_1, weight):
    alpha_1, beta_1 = result_1.solution
    predicted_1 = alpha_1 * weight + beta_1
    return (predicted_1,)


@app.cell
def _(mpg, plt, predicted_1, weight):
    plt.scatter(weight, mpg, label='data')
    plt.scatter(weight, predicted_1, label="predicted")
    plt.xlabel("weights")
    plt.ylabel("mpg")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_1, weight):
    plt.scatter(weight, predicted_1 - mpg, label="prediction error")
    plt.xlabel("weights")
    plt.ylabel("mpg error")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_1, sns):
    sns.displot(predicted_1 - mpg, stat="probability")
    plt.xlabel("error")
    return


@app.cell
def _(error_table, mpg, predicted_1):
    error_table(mpg, predicted_1)
    return


@app.cell
def _(mpg, pd, predicted_1, sns):
    sns.pairplot(pd.DataFrame({"mpg": mpg, "predicted": predicted_1}))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Linear Model wrt Inverse of Weight
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Search for $\alpha, \, \beta \in \mathbb{R}$ that provides the best approximation

    $$
    \text{mpg} \approx \frac{\alpha}{\text{weight}} + \beta
    $$

    Understood as, find the minimizer $\alpha$ of the loss function:

    $$
    \text{loss}(\alpha) =  \sum_k \left(\text{mpg}[k] - \frac{\alpha}{\text{weight}[k]} - \beta \right)^2.
    $$
    """)
    return


@app.cell
def _(mpg, ones, torch, weight):
    inv_weight = 1.0 / weight
    result_2 = torch.linalg.lstsq(torch.stack((inv_weight, ones), dim=1), mpg)
    result_2
    return inv_weight, result_2


@app.cell
def _(inv_weight, result_2):
    alpha_2, beta_2 = result_2.solution
    predicted_2 = alpha_2 * inv_weight + beta_2
    return (predicted_2,)


@app.cell
def _(mpg, plt, predicted_2, weight):
    plt.scatter(weight, mpg, label="data")
    plt.scatter(weight, predicted_2, label="predicted")
    plt.xlabel("weights")
    plt.ylabel("mpg")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_2, weight):
    plt.scatter(weight, predicted_2 - mpg, label="prediction error")
    plt.xlabel("weights")
    plt.ylabel("mpg error")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_2, sns):
    sns.displot(predicted_2 - mpg, stat="probability")
    plt.xlabel("error")
    return


@app.cell
def _(error_table, mpg, predicted_2):
    error_table(mpg, predicted_2)
    return


@app.cell
def _(mpg, pd, predicted_2, sns):
    sns.pairplot(pd.DataFrame({"mpg": mpg, "predicted": predicted_2}))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Take into Account Model Year
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Search for $\alpha, \, \beta, \, \gamma \in \mathbb{R}$ that provides the best approximation

    $$
    \text{mpg} \approx \frac{\alpha}{\text{weight}} + \beta +\gamma \times \text{year}
    $$

    Understood as, find the minimizer $\alpha$ of the loss function:

    $$
    \text{loss}(\alpha) =  \sum_k \left(\text{mpg}[k] - \frac{\alpha}{\text{weight}[k]} - \beta - \gamma \times \text{year}[k]\right)^2.
    $$
    """)
    return


@app.cell
def _(df, sns):
    sns.pairplot(df[["weight", "model_year", "mpg"]])
    return


@app.cell
def _(df, inv_weight, mpg, ones, torch):
    year = torch.tensor(df["model_year"])
    result_3 = torch.linalg.lstsq(torch.stack((inv_weight, ones, year), dim=1), mpg)
    result_3
    return result_3, year


@app.cell
def _(inv_weight, result_3, year):
    alpha_3, beta_3, gamma_3 = result_3.solution
    predicted_3 = alpha_3 * inv_weight + beta_3 + gamma_3 * year
    return (predicted_3,)


@app.cell
def _(mpg, plt, predicted_3, weight):
    plt.scatter(weight, mpg, label='data')
    plt.scatter(weight, predicted_3, label="predicted")
    plt.xlabel("weights")
    plt.ylabel("mpg")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_3, weight):
    plt.scatter(weight, predicted_3 - mpg, label="prediction error")
    plt.xlabel("weights")
    plt.ylabel("mpg error")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_3, sns):
    sns.displot(predicted_3 - mpg, stat="probability")
    plt.xlabel("error")
    return


@app.cell
def _(error_table, mpg, predicted_3):
    error_table(mpg, predicted_3)
    return


@app.cell
def _(mpg, pd, predicted_3, sns):
    sns.pairplot(pd.DataFrame({"mpg": mpg, "predicted": predicted_3}))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use the Origin
    """)
    return


@app.cell
def _(df, sns):
    sns.pairplot(df[["weight", "model_year", "mpg", "origin"]], hue="origin")
    return


@app.cell
def _(df, torch):
    from_USA = torch.tensor(df["origin"] == "USA", dtype=torch.float64)
    from_Europe = torch.tensor(df["origin"] == "Europe", dtype=torch.float64)
    from_Asia = torch.tensor(df["origin"] == "Asia", dtype=torch.float64)
    return from_Asia, from_Europe, from_USA


@app.cell
def _(from_Asia, from_Europe, from_USA, inv_weight, mpg, torch, year):
    result_4 = torch.linalg.lstsq(
        torch.stack((inv_weight, year, from_USA, from_Europe, from_Asia), dim=1),
        mpg,
    )
    result_4
    return (result_4,)


@app.cell
def _(result_4):
    alpha_4, gamma_4, beta_4_USA, beta_4_Europe, beta_4_Asia = result_4.solution
    return alpha_4, beta_4_Asia, beta_4_Europe, beta_4_USA, gamma_4


@app.cell
def _(
    alpha_4,
    beta_4_Asia,
    beta_4_Europe,
    beta_4_USA,
    from_Asia,
    from_Europe,
    from_USA,
    gamma_4,
    weight,
    year,
):
    predicted_4 = alpha_4 / weight + gamma_4 * year + beta_4_USA * from_USA + beta_4_Europe * from_Europe + beta_4_Asia * from_Asia
    return (predicted_4,)


@app.cell
def _(mpg, plt, predicted_4, weight):
    plt.scatter(weight, mpg, label="data")
    plt.scatter(weight, predicted_4, label="predicted")
    plt.xlabel("weights")
    plt.ylabel("mpg")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_4, weight):
    plt.scatter(weight, predicted_4 - mpg, label="prediction error")
    plt.xlabel("weights")
    plt.ylabel("mpg error")
    plt.legend()
    return


@app.cell
def _(mpg, plt, predicted_4, sns):
    sns.displot(predicted_4 - mpg, stat="probability")
    plt.xlabel("error")
    return


@app.cell
def _(error_table, mpg, predicted_4):
    error_table(mpg, predicted_4)
    return


@app.cell
def _(mpg, pd, predicted_4, sns):
    sns.pairplot(pd.DataFrame({"mpg": mpg, "predicted": predicted_4}))
    return


if __name__ == "__main__":
    app.run()
