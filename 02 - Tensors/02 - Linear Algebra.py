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
    # Linear Algebra

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Learning Objectives
    - [ ] Do basic linear algebra with the "vectors are matrices" mindset,
    - [ ] Understand the more idiomatic "vectors are 1d-tensors" approach,
    - [ ] Understand what tensor contraction does and how to use it,
    - [ ] Practice extra linear algebra tools: norms, inversion, eigenvalues/vectors, svd, lstsqr, etc..
    ///
    """
    )
    return


@app.cell
def _():
    import torch
    from torch import tensor
    return tensor, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Vectors""")
    return


@app.cell
def _(tensor):
    u = tensor([[1.0], 
                [2.0]])
    u
    return (u,)


@app.cell
def _(u):
    alpha = 2.0
    alpha * u
    return


@app.cell
def _(tensor):
    v = tensor([[3.0], 
                [4.0]])
    v
    return (v,)


@app.cell
def _(u, v):
    u + v
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Matrix-Vector Product""")
    return


@app.cell
def _(tensor):
    A = tensor([[1.0, 2.0],
                [3.0, 4.0]])
    A
    return (A,)


@app.cell
def _(tensor):
    x = tensor([[1.0], [2.0]])
    x
    return (x,)


@app.cell
def _(A, x):
    A * x # Not what you expect AT ALL!
    return


@app.cell
def _(A, x):
    y = A @ x
    y
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Matrix Product""")
    return


@app.cell
def _(tensor):
    B = tensor([[0.0, 1.0],
                [1.0, 0.0]])
    B
    return (B,)


@app.cell
def _(A, B):
    A * B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// warning
    `A * B` does **not** compute a the matrix product of `A` and `B`. 

    Instead it provides but the [Hadamard (or elementwise) product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) of `A` and `B`. 

    Use `A @ B` instead to to compute the matrix product of `A` and `B`.
    ///
    """
    )
    return


@app.cell
def _(A, B):
    A @ B
    return


@app.cell
def _(torch):
    C = torch.rand((2, 3))
    C
    return (C,)


@app.cell
def _(torch):
    D = torch.rand((3, 4))
    return (D,)


@app.cell
def _(C, D):
    E = C @ D
    E
    return (E,)


@app.cell
def _(E):
    E.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Transposition""")
    return


@app.cell
def _(x):
    x
    return


@app.cell
def _(x):
    x.T
    return


@app.cell
def _(A):
    A
    return


@app.cell
def _(A):
    A.T
    return


@app.cell
def _(C):
    C
    return


@app.cell
def _(C):
    C.T
    return


@app.cell
def _(A, x):
    x.T @ A
    return


@app.cell
def _(x, y):
    x.T @ y
    return


@app.cell
def _(x, y):
    (x.T @ y)[0, 0]
    return


@app.cell
def _(x, y):
    (x.T @ y).squeeze()
    return


@app.cell
def _(x, y):
    (x.T @ y).item()
    return


@app.cell
def _(A, x):
    x.T @ A @ x
    return


@app.cell
def _(x):
    x @ x.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Tensorisation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note

    The previous "describe vectors as matrices" trope works but is not idiomatic.
    They may described more easily as as 1-d tensors and basic linear algebra operations can still be carried out.
    ///
    """
    )
    return


@app.cell
def _(x):
    x
    return


@app.cell
def _(x):
    x.shape
    return


@app.cell
def _(x):
    x.squeeze()
    return


@app.cell
def _(tensor):
    x_1 = tensor([1.0, 2.0])
    return (x_1,)


@app.cell
def _(A, x_1):
    y_1 = A @ x_1
    y_1
    return


@app.cell
def _(A, B):
    A @ B
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// warning
    Transposition of a 1d vector is meaningless!
    """
    )
    return


@app.cell
def _(x_1):
    x_1.T
    return


@app.cell
def _(A, x_1):
    x_1 @ A @ x_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The operator `@` has a bunch of special cases in its [definition](https://pytorch.org/docs/stable/generated/torch.matmul.html). The situation is simpler when you consider `torch.tensordot` (the tensor contraction) instead.""")
    return


@app.cell
def _(torch):
    td = torch.tensordot
    return (td,)


@app.cell
def _(A, td, x_1):
    y_2 = td(A, x_1, 1)
    return (y_2,)


@app.cell
def _(A, B, td):
    td(A, B, 1)
    return


@app.cell
def _(td, x_1, y_2):
    td(y_2, x_1, 1)
    return


@app.cell
def _(td, x_1):
    td(x_1, x_1, dims=0)
    return


@app.cell
def _(td, x_1):
    td(x_1, x_1, dims=1)
    return


@app.cell
def _(A, B, td):
    td(A, B, dims=0) # Contract 0 dimensions
    return


@app.cell
def _(A, B, td):
    td(A, B, dims=1) # Contract 1 dimension
    return


@app.cell
def _(A, B, td):
    td(A, B, dims=2) # Contract 2 dimensions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Linear Algebra Operations""")
    return


@app.cell
def _():
    import torch.linalg as la
    return (la,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Norms""")
    return


@app.cell
def _(A):
    A
    return


@app.cell
def _(x_1):
    x_1
    return


@app.cell
def _(la, x_1):
    la.norm(x_1)
    return


@app.cell
def _(A, la):
    la.norm(A)
    return


@app.cell
def _(A, la):
    la.norm(A.flatten())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    See also: [vector_norm]  and [matrix_norm].

    [vector_norm]: https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html#torch.linalg.vector_norm

    [matrix_norm]: https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm
    """
    )
    return


@app.cell
def _(A, la):
    la.inv(A)
    return


@app.cell
def _(mo):
    mo.md(r"""### Inversions""")
    return


@app.cell
def _(A, la, x_1):
    la.inv(A) @ x_1
    return


@app.cell
def _(A, la, x_1):
    la.solve(A, x_1)
    return


@app.cell
def _(mo):
    mo.md(r"""### Eigenvectors and Eigenvalues""")
    return


@app.cell
def _(A, la):
    vals, vectors = la.eig(A)
    return vals, vectors


@app.cell
def _(vals):
    vals
    return


@app.cell
def _(vectors):
    vectors
    return


@app.cell
def _(A, la):
    la.eig(A)
    return


@app.cell
def _(A, la):
    vals_1, vects = la.eig(A)
    return vals_1, vects


@app.cell
def _(la, torch, vals_1, vects):
    vects @ torch.diag(vals_1) @ la.inv(vects)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There is a better function if the matrix is symmetric.""")
    return


@app.cell
def _(A):
    S_1 = A.T @ A
    S_1
    return (S_1,)


@app.cell
def _(S_1, la):
    la.eigh(S_1)
    return


@app.cell
def _(S_1, la, torch):
    vals_2, vects_1 = la.eigh(S_1)
    vects_1 @ torch.diag(vals_2) @ vects_1.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Singular Value Decomposition""")
    return


@app.cell
def _(A, la):
    la.svd(A)
    return


@app.cell
def _(A, la):
    U, S, Vh = la.svd(A)
    return S, U, Vh


@app.cell
def _(S, U, Vh, torch):
    U @ torch.diag(S) @ Vh
    return


@app.cell
def _(U):
    U.T @ U
    return


@app.cell
def _(Vh):
    Vh.T @ Vh
    return


if __name__ == "__main__":
    app.run()
