import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # MPG Dataset

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | Learning Objectives
    - [ ] Fetch the data set (use API and CSV to learn both ways),
    - [ ] Explore the metadata, interpret the info,
    - [ ] Make a dataframe from the data,
    - [ ] Make a seaborn pairplot with the suitable hue,
    - [ ] Clean-up: manage NaNs, fix errors, fix data types, get the brands, etc.
    - [ ] Save your result in a suitable format.

    """
    )
    return


@app.cell
def _():
    import csv
    import pathlib
    return (pathlib,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
    import seaborn as sns; sns.set_theme()
    import ucimlrepo
    return pd, requests, sns, ucimlrepo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataset""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ℹ️ Repository: [Auto MPG - UCI Machine Learning Repository]

    ℹ️ Data URL: <https://archive.ics.uci.edu/static/public/9/data.csv>

    🇺🇸 MPG stands for means "Miles Per Gallon"

    [Auto MPG - UCI Machine Learning Repository]: https://archive.ics.uci.edu/dataset/9/auto+mpg
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### With `ucimlrepo` and `pandas`""")
    return


@app.cell
def _(ucimlrepo):
    auto_mpg = ucimlrepo.fetch_ucirepo(id=9) 
    auto_mpg
    return (auto_mpg,)


@app.cell
def _(auto_mpg):
    list(auto_mpg.keys())
    return


@app.cell
def _(auto_mpg):
    metadata = auto_mpg["metadata"]
    metadata
    return


@app.cell
def _(auto_mpg):
    variables = auto_mpg["variables"]
    variables
    return


@app.cell
def _(auto_mpg):
    data = auto_mpg["data"]
    list(data.keys())
    return (data,)


@app.cell
def _(data):
    data["ids"]
    return


@app.cell
def _(data):
    X = data["features"]
    X
    return


@app.cell
def _(data):
    y = data["targets"]
    y
    return


@app.cell
def _(data):
    df = data["original"]
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Alternative: From the CSV file (with pandas)""")
    return


@app.cell
def _(df, pathlib, pd, requests):
    r = requests.get("https://archive.ics.uci.edu/static/public/9/data.csv")
    raw_data = r.content
    assert type(raw_data) == bytes
    pathlib.Path("tmp").mkdir(exist_ok=True)
    with open("tmp/auto_mpg.csv", mode="bw") as file:
        file.write(raw_data)
    df_ = pd.read_csv("tmp/auto_mpg.csv")
    assert all(df == df_)
    df_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Cleanup""")
    return


@app.cell
def _(df):
    df.isna().any()
    return


@app.cell
def _(df):
    df_1 = df.dropna().reset_index(drop=True)
    df_1
    return (df_1,)


@app.cell
def _(df_1):
    df_2 = df_1.copy()
    car_names = df_1["car_name"]
    brands = [car_name.split(",")[0].strip().capitalize() for car_name in car_names]
    models = [" ".join(car_name.split(",")[1:]).capitalize() for car_name in car_names]
    del df_2["car_name"]
    df_2.insert(0, "brand", brands)
    df_2.insert(1, "model", models)
    df_2 = df_2.sort_values(by="brand", ignore_index=True)
    def tweak_brand(name):
        if len(name) <= 3: # Acronyms: Amc, Bmw, etc.
            name = name.upper()
        fixes = {
            "AMC": "American Motors",
            "HI": "International Harvester",
            "Chevroelt": "Chevrolet",
            "Chevy": "Chevrolet",
            "Toyouta": "Toyota",
            "Maxda": "Mazda",
            "Mercedes-benz": "Mercedes-Benz",
            "Mercedes": "Mercedes-Benz",
            "Vokswagen": "Volkswagen",
            "VW": "Volkswagen",

        }
        name = fixes.get(name) or name
        return name
    df_2["brand"] = df_2["brand"].apply(tweak_brand)
    df_2 = df_2.sort_values(by="brand", ignore_index=True)

    df_2["brand"].unique()
    return (df_2,)


@app.cell
def _(df_2):
    df_2
    return


@app.cell
def _(df_2):
    df_2["origin"].unique()
    return


@app.cell
def _(df_2):
    df_2[["origin", "brand"]].sort_values(by="origin", ignore_index=True)
    return


@app.cell
def _(df_2):
    df_3 = df_2.copy()
    def convert_origin(number):
        return {1: "USA", 2: "Europe", 3: "Asia"}.get(number)
    df_3["origin"] = df_3["origin"].apply(convert_origin)
    df_3
    return (df_3,)


@app.cell
def _(df_3, sns):
    sns.pairplot(df_3[["model_year", "weight", "origin", "mpg"]], hue='origin')
    return


@app.cell
def _(df_3):
    df_3
    return


@app.cell
def _(df_3):
    # ⚠️ These changes will be lost if df_4 is exported as csv ... but not parquet! 🥳
    df_4 = df_3.copy()
    df_4["origin"] = df_3["origin"].astype("category")
    df_4["weight"] = df_4["weight"].astype(float)
    df_4
    return (df_4,)


@app.cell
def _(df_4):
    df_4.to_csv("data/auto_mpg.csv", index=False)
    return


@app.cell
def _(pd):
    pd.read_csv("data/auto_mpg.csv")
    return


@app.cell
def _(df_4):
    df_4.to_parquet("data/auto_mpg.parquet")
    return


@app.cell
def _(pd):
    pd.read_parquet("data/auto_mpg.parquet")
    return


if __name__ == "__main__":
    app.run()
