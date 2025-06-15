import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Tensor Algebra

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell
def _():
    import torch
    from torch import tensor, tensordot, einsum
    return einsum, tensordot, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tensor product & contraction

    | Tensor Operation      | Symbol    | Implementation                |
    | --------------------- | --------- | ----------------------------- |
    | Product               | $\otimes$ | `tensordot(..., ..., ndim=0)` |
    | Contraction           | $\cdot$   | `tensordot(..., ..., ndim=1)` |
    | Double Contraction    | $:$       | `tensordot(..., ..., ndim=2)` |
    | ...                   | ...       | ...                           |
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(torch):
    torch.manual_seed(42)

    a = torch.rand(size=())
    b = torch.rand(size=())
    x = torch.rand(size=(2, ))
    y = torch.rand(size=(2, ))
    R = torch.rand(size=(2, 2,))
    S = torch.rand(size=(2, 2,))
    return R, S, a, b, x, y


@app.cell(hide_code=True)
def _(R, S, a, b, mo, x, y):
    mo.md(
        rf"""
    | Variable Name      | Value                          | Rank     | Shape              |
    | ------------------ | ------------------------------ | -------- | ------------------ |
    | a                  | {a!r}                          | {a.ndim} | {tuple(a.shape)}   |
    | b                  | {b!r}                          | {b.ndim} | {tuple(b.shape)}   |
    | x                  | {x!r}                          | {x.ndim} | {tuple(x.shape)}   |
    | y                  | {y!r}                          | {y.ndim} | {tuple(y.shape)}   |
    | R                  | {repr(R).replace("\n", " ")}   | {R.ndim} | {tuple(R.shape)}   |
    | S                  | {repr(S).replace("\n", " ")}   | {S.ndim} | {tuple(S.shape)}   |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    ### Tensor product of scalars

    $$
    [a] \otimes [b] = [a \times b] 
    $$
    """
    )
    return


@app.cell
def _(a, b, tensordot):
    tensordot(a, b, dims=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Tensor product of 0d and 1d tensors


    $$
    [a] \otimes [x_i]_i = [x_i]_i \otimes [a] = [a \times x_i]_i
    $$
    """
    )
    return


@app.cell
def _(a, tensordot, x):
    tensordot(a, x, dims=0)
    return


@app.cell
def _(a, tensordot, x):
    tensordot(x, a, dims=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Tensor product of 1d tensors 

    $$
    [x_i]_i \otimes [y_j]_j = [x_i \times y_i]_{ij}
    $$
    """
    )
    return


@app.cell
def _(tensordot, x, y):
    tensordot(x, y, dims=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    ### Tensor contraction of 1d tensors 


    $$
    [x_i]_i \cdot [y_j]_j = \left[\sum_k x_k y_k\right]
    $$
    """
    )
    return


@app.cell
def _(tensordot, x, y):
    tensordot(x, y, dims=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Tensor product of of 0d and 1d tensors 

    $$
    [a] \otimes [R_{ij}]_{ij} = [R_{ij}]_{ij} \otimes [a] = [a \otimes R_{ij}]_{ij}
    $$
    """
    )
    return


@app.cell
def _(R, a, tensordot):
    tensordot(a, R, dims=0) 
    return


@app.cell
def _(R, a, tensordot):
    tensordot(R, a, dims=0)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Tensor product of 1d and 2d tensors

    $$
    [R_{ij}]_{ij} \otimes [x_k]_k= [R_{ij} x_k]_{ijk}
    $$

    $$
    [x_i]_i \otimes [R_{jk}]_{jk} = [x_i R_{jk}]_{ijk}
    $$
    """
    )
    return


@app.cell
def _(R, torch, x):
    torch.tensordot(R, x, dims=0)
    return


@app.cell
def _(R, torch, x):
    torch.tensordot(R, x, dims=0)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Tensor contraction of 1d and 2d tensor

    $$
    [R_{ij}]_{ij} \cdot [x_k]_k= \left[\sum_j R_{ij} x_j\right]_{i}
    $$

    $$
    [x_i]_i \cdot [R_{jk}]_{jk} = \left[\sum_l x_l R_{lk}\right]_{k}
    $$
    """
    )
    return


@app.cell
def _(R, torch, x):
    torch.tensordot(R, x, dims=1)
    return


@app.cell
def _(R, torch, x):
    torch.tensordot(x, R, dims=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Tensor product of 2d tensors

    $$
    [R]_{ij} \otimes [S]_{kl} = [R_{ij} S_{kl}]_{ijkl}
    $$
    """
    )
    return


@app.cell
def _(R, S, torch):
    torch.tensordot(R, S, dims=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Tensor contraction of 2d tensors

    $$
    [R_{ij}]_{ij} \cdot [S_{kl}]_{kl} = \left[\sum_m R_{im} S_{ml} \right]_{il}
    $$
    """
    )
    return


@app.cell
def _(R, S, torch):
    torch.tensordot(R, S, dims=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Double contraction of 2d tensors

    $$
    [R_{ij}]_{ij} : [S_{kl}]_{kl} = \left[\sum_{m, n} R_{mn} S_{mn} \right]
    $$
    """
    )
    return


@app.cell
def _(R, S, torch):
    torch.tensordot(R, S, dims=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// warning | Shape compatibility

    The tensor product of two tensors $T_1$ and $T_2$ is defined unconditionally but their simple and double contraction are only defined when their shapes are compatible:

    | Operation          | Shape of $T_1$      | Shape of $T_2$      | Compatibility conditions | Shape of the result                  |
    | ------------------ | ------------------- | ------------------- | ------------------------ | -------------------------------------- | 
    | $T_1 \otimes T_2$  | $(i_1, \dots, i_k)$ | $(j_1, \dots, j_l)$ |                          | $(i_1, \cdots, i_k, j_1, \dots, j_k)$  |
    | $T_1 \cdot T_2$    | $(i_1, \dots, i_k)$ | $(j_1, \dots, j_l)$ | $i_k=j_1$                | $(i_1, \cdots, i_{k-1}, j_2, \dots, j_k)$  |
    | $T_1 : T_2$        | $(i_1, \dots, i_k)$ | $(j_1, \dots, j_l)$ | $i_{k-1} = j_1$ and $i_k=j_2$ | $(i_1, \cdots, i_{k-2}, j_3, \dots, j_k)$      |

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | What about the `@` operator?

    `T_1 @ T_2` is equivalent to `tensordot(T_1, T_2)` when `T_1` and `T_2` are 1d or 2d tensors.

    For higher ranks, extra dimensions are interpreted as batch dimensions and the behavior is therefor different; refer to [torch.matmul] for more details.

    [torch.matmul]: https://docs.pytorch.org/docs/main/generated/torch.matmul.html

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Einstein Summation

    Similar to `tensordot`, but even more general.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    [x_i]_i \to [x_i]_i
    $$
    """
    )
    return


@app.cell
def _(einsum, x):
    einsum("i->i", x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    [x_i]_i \to \left[\sum_i x_i\right]
    $$
    """
    )
    return


@app.cell
def _(einsum, x):
    einsum("i->", x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Scalar product

    $$
    x \otimes y = [x_i y_j]_{ij}
    $$
    """
    )
    return


@app.cell
def _(einsum, x, y):
    einsum("i,j->ij", x, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    x \cdot y = \left[ \sum_k x_k y_k\right]
    $$
    """
    )
    return


@app.cell
def _(einsum, x, y):
    einsum("i,j->", x, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Transposition

    $$
    [R_{ij}]_{ij} \to [R_{ji}]_{ij} 
    $$
    """
    )
    return


@app.cell
def _(R, einsum):
    einsum("ij->ji", R)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Trace

    $$
    [R_{ij}]_{ij} \to \left[\sum_{k} R_{kk} \right]
    $$
    """
    )
    return


@app.cell
def _(R, einsum):
    einsum("ij->", R)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    R \otimes S = [R_{ij}S_{kl}]_{ijkl}
    $$
    """
    )
    return


@app.cell
def _(R, S, torch):
    torch.einsum("ij,kl->ijkl", R, S)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Matrix Product
    $$
    R \cdot S = \left[ \sum_m R_{im} S_{ml}\right]_{il}
    $$
    """
    )
    return


@app.cell
def _(R, S, torch):
    torch.einsum("ij,kl->il", R, S)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    R : S = \left[ \sum_{m, n} R_{mn} S_{mn}\right]
    $$
    """
    )
    return


@app.cell
def _(R, S, torch):
    torch.einsum("ij,kl->", R, S)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Batched Matrix Product""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Consider (2d) batches of matrices of size $(4,5)$ and $(5,6)$:""")
    return


@app.cell
def _(torch):
    A = torch.rand(2, 3, 4, 5)
    B = torch.rand(2, 3, 5, 6)
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can compute the batches of products with:""")
    return


@app.cell
def _(A, B, torch):
    torch.einsum('abij,abjk->abik', A, B)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The ellipsis `...` provide a convenient shortcut for the batched dimensions.""")
    return


@app.cell
def _(A, B, torch):
    torch.einsum('...ij,...jk->...ik', A, B)
    return


if __name__ == "__main__":
    app.run()
