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
    # Tensors

    [Sébastien Boisgérault], [ITN], Mines Paris - PSL University

    [ITN]: https://itn.dev
    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Getting Started""")
    return


@app.cell
def _(torch):
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return


@app.cell
def _(torch):
    torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    return


@app.cell
def _(torch):
    torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
            ],
            [
                4.0,
                5.0,
                6.0,
            ],
        ]
    )
    return


@app.cell
def _(torch):
    t = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    return (t,)


@app.cell
def _(t):
    t.ndim
    return


@app.cell
def _(t):
    t.shape
    return


@app.cell
def _(t):
    t.dtype
    return


@app.cell
def _(t):
    bytes(t.numpy())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""-----""")
    return


@app.cell
def _(torch):
    t0 = torch.tensor(1.0)
    t0
    return (t0,)


@app.cell
def _(torch):
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t1
    return (t1,)


@app.cell
def _(torch):
    t2 = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    t2
    return (t2,)


@app.cell
def _(torch):
    t3 = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            [
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
        ]
    )
    t3
    return (t3,)


@app.cell(hide_code=True)
def _(pd, t0, t1, t2, t3):
    shapes = []
    for i, _t in enumerate([t0, t1, t2, t3]):
        shapes.append(
            {
                "name": f"t{i}",
                "ndim": _t.ndim,
                "shape": _t.shape,
            }
        )
    pd.DataFrame(shapes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Tensor Creation""")
    return


@app.cell
def _(torch):
    torch.empty([2, 3])
    return


@app.cell
def _(torch):
    torch.zeros([2, 3])
    return


@app.cell
def _(torch):
    torch.ones([2, 3])
    return


@app.cell
def _(torch):
    torch.rand([2, 3])
    return


@app.cell
def _(torch):
    torch.manual_seed(42)
    torch.rand([2, 3])
    return


@app.cell
def _(torch):
    torch.manual_seed(42)
    torch.rand([2, 3])
    return


@app.cell
def _(torch):
    torch.normal(0, 1, [2, 3])
    return


@app.cell
def _(torch):
    torch.arange(0, 100, 10)
    return


@app.cell
def _(torch):
    torch.linspace(0.0, 1.0, 11)
    return


@app.cell
def _(torch):
    torch.logspace(0, 8, steps=9, base=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Indexing""")
    return


@app.cell
def _(t1):
    t1
    return


@app.cell
def _(t2):
    t2
    return


@app.cell
def _(t2):
    t2[1, 2]
    return


@app.cell
def _(t3):
    t3
    return


@app.cell
def _(t3):
    t3[1, 1, 2]
    return


@app.cell
def _(t3):
    t3[-1, -1, -1]
    return


@app.cell
def _(t3):
    t3[1, 1, :]
    return


@app.cell
def _(t3):
    t3[0:2, 0:2, 0:3]
    return


@app.cell
def _(t3):
    t3[0, 0, 0:3]
    return


@app.cell
def _(t3):
    t3[0, 0:2, 0:3]
    return


@app.cell
def _(t3):
    t3[0,:,:]
    return


@app.cell
def _(t3):
    t3[0,...]
    return


@app.cell
def _(t3):
    t3 >= 4
    return


@app.cell
def _(t3):
    t3[t3>=4]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Vectorization""")
    return


@app.cell
def _(t0):
    t0
    return


@app.cell
def _(t0):
    1.0 + t0
    return


@app.cell
def _(t0):
    t0 + t0
    return


@app.cell
def _(t0):
    2.0 * t0
    return


@app.cell
def _(t0):
    t0 * t0
    return


@app.cell
def _(t0):
    t0.sin()
    return


@app.cell
def _(t0, torch):
    torch.sin(t0)
    return


@app.cell
def _(t1):
    t1
    return


@app.cell
def _(t1):
    1.0 + t1
    return


@app.cell
def _(t1):
    t1 + t1
    return


@app.cell
def _(t1):
    2.0 * t1
    return


@app.cell
def _(t1):
    t1 * t1
    return


@app.cell
def _(t1):
    t1.sin()
    return


@app.cell
def _(t1, torch):
    torch.sin(t1)
    return


@app.cell
def _(t2):
    t2
    return


@app.cell
def _(t2):
    1.0 + t2
    return


@app.cell
def _(t2):
    2 * t2
    return


@app.cell
def _(t2):
    t2 * t2
    return


@app.cell
def _(t2):
    t2.sin()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Reshaping""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Flatten, squeeze and reshape""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip
    The following operations on do not change the elements in a tensor or their order.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`flatten` makes a tensor 1-dimensional""")
    return


@app.cell
def _(t3):
    t3
    return


@app.cell
def _(t3):
    t3.shape
    return


@app.cell
def _(t3):
    t3.flatten()
    return


@app.cell
def _(t3):
    t3.flatten().shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can also selectively flatten a tensor along *some* of its dimensions:""")
    return


@app.cell
def _(t3):
    t3.flatten(0, 1)
    return


@app.cell
def _(t3):
    t3.flatten(0, 1).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`squeeze` remove the dimensions with only one element.""")
    return


@app.cell
def _(torch):
    t_2 = torch.tensor(
        [
            [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]],
        ]
    )
    t_2
    return (t_2,)


@app.cell
def _(t_2):
    t_2.shape
    return


@app.cell
def _(t_2):
    t_2.squeeze()
    return


@app.cell
def _(t_2):
    t_2.squeeze().shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can also select only some (trivial) dimensions to be removed:""")
    return


@app.cell
def _(t_2):
    t_2.squeeze(0)
    return


@app.cell
def _(t_2):
    t_2.squeeze(0).shape
    return


@app.cell
def _(t_2):
    t_2.squeeze(3)
    return


@app.cell
def _(t_2):
    t_2.squeeze(3).shape
    return


@app.cell
def _(t_2):
    t_2.squeeze((0, 3))
    return


@app.cell
def _(t_2):
    t_2.squeeze((0, 3)).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`flatten` and `squeeze` are special cases of `reshape`.""")
    return


@app.cell
def _(t3):
    t3.reshape(2*2*3)
    return


@app.cell
def _(t3):
    t3.reshape(2*2*3).shape
    return


@app.cell
def _(torch):
    t_3 = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]])
    t_3.shape
    return (t_3,)


@app.cell
def _(t_3):
    t_3.reshape([3, 2])
    return


@app.cell
def _(t_3):
    t_3.reshape([3, 2]).shape
    return


@app.cell
def _(t3):
    t3.reshape([3, 4])
    return


@app.cell
def _(t3):
    t3.reshape([1, 2, 6])
    return


@app.cell
def _(t3):
    t3.reshape([1, 2, 6]).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Since the product of all shape coefficients is the number of elements and should be preserved, one of the coefficients can always be automatically deduced; use `-1` to let pytorch compute it.""")
    return


@app.cell
def _(t3):
    t3.reshape([1, 2, -1]).shape
    return


@app.cell
def _(t3):
    t3.reshape([-1]).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Transposition and Permutation""")
    return


@app.cell
def _(t2):
    t2
    return


@app.cell
def _(t2):
    t2.T
    return


@app.cell
def _(t2):
    t2.transpose(0, 1)
    return


@app.cell
def _(t3):
    t3
    return


@app.cell
def _(t3):
    t3.shape
    return


@app.cell
def _(t3):
    t3.transpose(0, 1)
    return


@app.cell
def _(t3):
    t3.transpose(0, 1).shape
    return


@app.cell
def _(t3):
    t3.transpose(0, 2)
    return


@app.cell
def _(t3):
    t3.transpose(0, 2).shape
    return


@app.cell
def _(t3):
    t3.permute([2, 1, 0])
    return


@app.cell
def _(t3):
    t3.permute([0, 2, 1])
    return


@app.cell
def _(t3):
    t3.permute([0, 2, 1]).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Concatenation and Stacking""")
    return


@app.cell
def _(torch):
    A = torch.tensor([[1, 2, 3], [4, 5, 6]])
    B = torch.tensor([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    torch.cat((A, B))
    return A, B


@app.cell
def _(A, B, torch):
    C = torch.cat((A, B))
    D = C[:, 0:1]
    E = C[:, 1:]
    torch.cat((D, E), dim=1)
    return


@app.cell
def _(torch):
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([3.0, 4.0, 5.0])
    s = torch.stack((x, y))
    s
    return s, x, y


@app.cell
def _(s):
    s.shape
    return


@app.cell
def _(torch, x, y):
    s_1 = torch.stack((x, y), dim=0)
    s_1.shape
    return


@app.cell
def _(torch, x, y):
    s_2 = torch.stack((x, y), dim=1)
    s_2.shape
    return


@app.cell
def _(torch):
    A_2 = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    B_2 = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    return A_2, B_2


@app.cell
def _(A_2, B_2, torch):
    torch.stack((A_2, B_2))
    return


@app.cell
def _(A_2, B_2, torch):
    torch.stack((A_2, B_2), dim=0)
    return


@app.cell
def _(A_2, B_2, torch):
    torch.stack((A_2, B_2), dim=1)
    return


@app.cell
def _(A_2, B_2, torch):
    torch.stack((A_2, B_2), dim=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Expansion and Broadcasting""")
    return


@app.cell
def _(torch):
    t_4 = torch.tensor(
        [
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[5.0], [6.0]],
            ],
        ]
    )
    t_4
    return (t_4,)


@app.cell
def _(t_4):
    t_4.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Duplicate the elements along (some of)  the dimensions with a depth of 1:""")
    return


@app.cell
def _(t_4):
    t_4.expand([1, 3, 2, 2])
    return


@app.cell
def _(t_4):
    t_4.expand([1, 3, 2, 2]).shape
    return


@app.cell
def _(t_4):
    t_4.expand([3, 3, 2, 2])
    return


@app.cell
def _(t_4):
    t_4.expand([3, 3, 2, 2]).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can also prepend new dimensions:""")
    return


@app.cell
def _(t_4):
    t_4.expand([3, 1, 3, 2, 1])
    return


@app.cell
def _(t_4):
    t_4.expand([3, 1, 3, 2, 1]).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Some binary operations automatically perform the appropriate expansions on their arguments if their shapes are somehow consistent or *broadcastable*; [the rules are](https://pytorch.org/docs/stable/notes/broadcasting.html#general-semantics):


    /// note | Broadcasting Rules
    Two tensors are *broadcastable* if the following rules hold:

    - Each tensor has at least one dimension.

    - When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Example with the addition:""")
    return


@app.cell
def _(torch):
    r = torch.rand([2, 1, 3])
    return (r,)


@app.cell
def _(r):
    r
    return


@app.cell
def _(torch):
    s_3 = torch.rand([3, 2, 2, 1])
    s_3
    return (s_3,)


@app.cell
def _(r, s_3):
    r + s_3
    return


@app.cell
def _(r, s_3):
    (r + s_3).shape
    return


if __name__ == "__main__":
    app.run()
