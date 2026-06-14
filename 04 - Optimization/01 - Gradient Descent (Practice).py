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
    # Gradient Descent


    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Learning Objectives
    - [ ] Principle of gradient descent optimization
    - [ ] Manual implementation of the algorithm
    - [ ] Experiment with learning rate tuning with quadratic functions
    - [ ] Reimplementation of gradient descent with `PyTorch optim.SGD`.
    ///
    """)
    return


@app.cell
def _():
    import torch
    import torch.func
    import torch.linalg
    import torch.optim

    return (torch,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Principles of Gradient Descent

    Let $f: \mathbb{R}^d \to \mathbb{R}$ be a differentiable function and let $x_0 \in \mathbb{R}^d$.

    Let $\eta >0$; the Taylor expansion of $f$ at $x$ provides:

    \begin{align*}
    f\left(x_0 - \eta \nabla f(x) \right)
      &= f(x_0) +  \nabla f(x_0) \cdot  (-\eta \nabla f(x_0)) + o(\eta) \\
      &= f(x_0) - \eta \nabla f(x_0) \cdot \nabla f(x_0) + o(\eta) \\
      &= f(x_0) - \eta \|\nabla f(x_0)\|^2 + o(\eta).
    \end{align*}

    So if $\nabla f(x_0) \neq 0$, for any $\eta>0$ **small enough**, the term $\eta \|\nabla f(x_0)\|^2$ is larger than $o(\eta)$ and if we define

    $$
    x_1 := x_0 - \eta \nabla f(x_0)
    $$

    then by construction

    $$
    f(x_1) < f(x_0).
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Gradient Descent Minimization


    0. **Learning rate.** Select a small $\eta > 0$, for example $10^{-3}$,


    1. **Initial guess.** Pick $x_0 \in \mathbb{R}^d$ at random (or make a guess!).


    2. **Iterations.** Compute repeatedly $x_{n+1} := x_n - \eta \nabla f(x_{n})$,

    3. **Termination.** Stop when progress stalls (or your computation budget is over).


    Then your approximation of the minimizer $x_*$ of $f$ is

    $$x_* := x_n.$$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimization of Quadratic Functions

    Quadratic functions generalize 2-order polynomials

    $$
    f(x) = \frac{a}{2} x^2 + b x + c, \; x \in \mathbb{R}
    $$

    to variables in higher dimensions

    $$
    f(x) = x \cdot \frac{A}{2} \cdot x + b \cdot x + c, \; x \in \mathbb{R}^d
    $$

    where $A \in \mathbb{R}^{d\times d}$, $b \in \mathbb{R}^d$, $c \in \mathbb{R}^n$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Why this interest in quadratic functions?

    -   The minimisation of quadratic function solves the (ubiquitous!) linear regression problem: to find the weight matrix $A$ and bias $b$ such that $A x + b$ is the best prediction of $y$, solve

        $$
        \min_{A, b} \|A x + b - y \|^2.
        $$

        This criterion which is quadratic w.r.t. $(A, b)$[^1]!

    -   They approximate locally any (twice continuously differentiable) function at the order 2:

        $$
        f(x) = \frac{1}{2} (x - x_0) \cdot \nabla^2 f(x_*) \cdot (x - x_0) + \nabla f(x_0) \cdot (x - x_0) + f(x_0) + o(\|x-x_0\|^2)
        $$

    -   Many minimisation techniques of quadratic functions generalize to convex functions.

    ///

    [^1]: Suitably "flattened" as a vector of $\mathbb{R}^{d \times d + d}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Quadratic Functions in $\mathbb{R}^2$

    1.  Pick a random matrix $A \in \mathbb{R}^{2\times2}$, vector $b \in \mathbb{\R}^2$ and scalar $c \in \mathbb{R}$. Define in a PyTorch function $f$ that given $x \in \mathbb{R}^2$ returns the scalar

        $$
        f(x) = x \cdot \frac{A}{2} \cdot x + b \cdot x + c \in \mathbb{R}.
        $$

    2.  Check that $f$ is proper, that is

        $$
        f(x) \to +\infty \;\;\; \; \text{ when } \;\; \|x\| \to + \infty.
        $$

        If that's not the case, repick $A$ until this property holds!


        (Every proper quadratic function has a single minimum.)



    ///
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(42)
    A = torch.rand((2, 2))
    while torch.linalg.det(A) <= 0 or torch.trace(A) <= 0:
        A = torch.rand((2, 2))
    b = torch.rand((2,))
    c = torch.rand(())
    A, b, c
    return A, b, c


@app.cell
def _(A, b, c, torch):
    def fv1(x):
        return x @ (A / 2) @ x + b @ x + c

    fv1(torch.tensor([1.0, 2.0]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip |

    3.  Now provide a vectorized PyTorch implementation of $f$, that will associate to any
        tensor $x$ of shape $(i_1, \dots, i_k, 2)$ the corresponding tensor
        $f(x)$ of shape $(i_1, \dots, i_k)$.

    4.  Use the [contour] function of matplotlib to represent level curves of your functions.
        Adjust the range of the input variables to display a neighborhood of the minimum.

    [contour]: (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)

    ///
    """)
    return


@app.cell(hide_code=True)
def _(A, b, c, torch):
    def f(x):
        return (
            torch.einsum("...i,ij,...j->...", x, A / 2 , x) + 
            torch.einsum("i,...i->...", b, x) +
            c
        )

    _x = torch.rand((3, 4, 2))
    f(_x)
    return (f,)


@app.cell(hide_code=True)
def _(f, plt, torch):
    _x = torch.linspace(-5.0, 5.0, 100)
    _y = torch.linspace(-5.0, 5.0, 100)
    _X, _Y = torch.meshgrid(_x, _y, indexing="ij")
    _Z = f(torch.stack((_X, _Y), dim=2))
    plt.contour(_X, _Y, _Z, levels=30)
    plt.grid(True)
    plt.colorbar()
    plt.gcf()
    return


@app.cell
def _(f, plt, torch):
    def contour(f, x, y, levels=30):
        _X, _Y = torch.meshgrid(x, y, indexing="ij")
        _Z = f(torch.stack((_X, _Y), dim=2))
        plt.contour(_X, _Y, _Z, levels=levels)
        plt.grid(True)
        plt.colorbar()
        return plt.gcf()


    contour(
        f,
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return (contour,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip |

    6.   Pick a random initial value of $x_0$ and a value $\eta$ of the learning rate (for example $10^{-3}$).
        Compute a gradient descent sequence with $n=100$ points. Display this sequence on top of the contour.

    7.  Change the value of the learning rate and experiment until you have discovered the learning rate at the limit of the convergence.
        Test the gradient sequence with half that value ; experiment with other values until you find (approximately) the learning rate that ensures the fastest convergence.

    ///
    """)
    return


@app.cell
def _(f, torch):
    def grad_descent(f, x_0, lr=1e-3, n=100):
        xs = [x_0]
        x = x_0
        for _ in range(n): 
            x = x.clone()
            x.requires_grad = True
            y = f(x)
            y.backward()
            grad_f_x = x.grad
            x = x.detach() - lr * grad_f_x
            xs.append(x)
        return xs

    grad_descent(f, x_0=torch.tensor([-4.0, -4.0]))
    return (grad_descent,)


@app.cell
def _(contour, mo, plt):
    def gradient_sequence_plot(f, xys, x, y):
        fig = contour(f, x, y)
        _xs = [_xy[0] for _xy in xys]
        _ys = [_xy[1] for _xy in xys]
        plt.plot(_xs, _ys, "r-")
        plt.plot(_xs[-1], _ys[-1], "r+")
        return mo.center(fig)

    return (gradient_sequence_plot,)


@app.cell
def _(f, grad_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=grad_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=1e-3,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell
def _(f, grad_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=grad_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=1e-2,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell
def _(f, grad_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=grad_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=1e-1,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell
def _(f, grad_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=grad_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=5e-1,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell
def _(f, grad_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=grad_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=6.3e-1,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell
def _(f, grad_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=grad_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=1.0,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell
def _(f, grad_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=grad_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=1.28,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell
def _(f, grad_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=grad_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=1.28 / 2,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip |

    8.   Replicate this computation using the PyTorch [`SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) optimizer class.
    """)
    return


@app.cell
def _(f, gradient_sequence_plot, torch):
    def _():
        x = torch.tensor([-4.0, 0.0], requires_grad=True)
        n = 100
        xs = []
        optimizer = torch.optim.SGD(params=[x], lr=5e-1)
        for _ in range(n):
            xs.append(x.detach().clone())
            optimizer.zero_grad()
            f(x).backward()
            optimizer.step()
        return gradient_sequence_plot(
            f=f,
            xys=xs,
            x=torch.linspace(-5.0, 5.0, 100),
            y=torch.linspace(-5.0, 5.0, 100),
        )
    _()
    return


if __name__ == "__main__":
    app.run()
