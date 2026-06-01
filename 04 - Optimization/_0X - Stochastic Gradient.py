import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// Warning | TODO
    - [ ] Explain/Show that with partial information, the input/output relationship is not a function really. The "right" model considers that input and output are random values (this is a strict superset of the function modelling)
    - [ ] Explain/Show that input dataset are necessarily partial and that anyway with huge dataset you can never load all the stuff at once in memory -> You do GD on a partial dataset. What does it change ? (Theory (from Bach), expectation of the loss, practice : convergence under good properties with decreasing learning rates.)
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
