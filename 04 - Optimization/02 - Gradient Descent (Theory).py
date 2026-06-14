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

    - [ ] Analysis of stability for quadratic functions
    - [ ] Analysis of speed of convergence for quadratic functions
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
        f(x) = x \cdot A \cdot x + b \cdot x + c \in \mathbb{R}.
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
    def f(x):
        return (
            torch.einsum("...i,ij,...j->...", x, A / 2, x) + 
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


@app.cell
def _(f, torch):
    def gradient_descent(f, x_0, lr=1e-3, n=100):
        xs = [x_0]
        x = x_0.clone(); x.requires_grad = True
        optimizer = torch.optim.SGD(params=[x], lr=lr)
        for _ in range(n):
            optimizer.zero_grad()
            f(x).backward()
            optimizer.step()
            xs.append(x.detach().clone())
        return xs

    gradient_descent(f, x_0=torch.tensor([-4.0, -4.0]))
    return (gradient_descent,)


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
def _(f, gradient_descent, gradient_sequence_plot, torch):
    gradient_sequence_plot(
        f=f,
        xys=gradient_descent(
            f,
            x_0=torch.tensor([-4.0, 0.0]),
            lr=0.2,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Theory

    ### Minimum

    Let $A \in \mathbb{R}^{d \times d}$, $b \in \mathbb{R}^d$, $c \in \mathbb{R}$ and
    $f :\mathbb{R}^d \to \mathbb{R}$ be the function defined by:

    $$
    f(x) := x \cdot \frac{A}{2} \cdot x + b \cdot x + c.
    $$

    Then, for any $x \in \mathbb{R}^d$ and $\Delta x \in \mathbb{R}^d$, we have

    $$
    \begin{split}
    df(x) (\Delta x)
      &= \Delta x \cdot  (A / 2) \cdot x + x \cdot (A / 2) \cdot \Delta x + b \cdot \Delta x \\
      &= x \cdot (A^T / 2) \cdot \Delta x + x \cdot (A / 2) \cdot \Delta x + b \cdot \Delta x \\
      &= (x\cdot (A^T / 2) + x\cdot (A / 2) + b) \cdot \Delta x \\
      &= ((A / 2) \cdot x + (A^T/2)\cdot x + b) \cdot \Delta x \\
      &= ((A + A^T)/2 \cdot x + b) \cdot \Delta x \\
    \end{split}
    $$

    Hence,

    $$
    \nabla f(x) = \frac{A + A^T}{2} \cdot x + b.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Matrix Symmetrization

    We call **symmetrization of $A$** the matrix

    $$
    \sigma(A):= \frac{1}{2}(A + A^T) \in \mathbb{R}^{d \times d}.
    $$

    By construction, the matrix $\sigma(A)$ is **symmetric**:

    $$
    \sigma(A)^T = \sigma(A).
    $$

    ///
    """)
    return


@app.cell
def _(A):
    sigma_A = (A + A.T) / 2
    sigma_A
    return (sigma_A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Symmetry Assumption
    Note that since

    $$
    \begin{split}
    f(x)
      &= x \cdot (A/2) \cdot x + b \cdot x + c \\
      &= \frac{1}{2} x \cdot (A/2) \cdot x + \frac{1}{2} x \cdot (A^T/2) \cdot x + b\cdot x + c \\
      & = \frac{1}{2} x \cdot \left((A + A^T \right)/2) x + b \cdot x + c \\
      &= x\cdot (\sigma(A)/2) x + b \cdot x + c
    \end{split}
    $$

    we may always replace $A$ with its symmetrization (or assume that $A$ was symmetric in the first place!).


    The computations we have performed previously yield

    $$
    \nabla f(x) = \sigma(A) x + b.
    $$
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Positive definitiveness

    A matrix $A \in \mathbb{R}^{d \times d}$ is **positive definite** when

    $$
    \forall \, x \in \mathbb{R}^n, \; x \cdot \sigma(A) \cdot x \geq 0
    \quad
    \text{ and }
    \quad
    x \cdot \sigma(A) \cdot x  = 0 \,\Rightarrow \, x = 0
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Existence, uniqueness and value of the minimum
    If $A$ is positive definite, then the function $f(x)$ has a unique global minimum.

    The minimal argument $x_*$ is the (unique) solution of

    $$
    \nabla f(x_*) = 0
    $$

    which is

    $$
    x_* = - \sigma(A)^{-1} \cdot b.
    $$

    and consequently

    $$
    f(x_*) = -\frac{1}{2} b \cdot \sigma(A)^{-1} \cdot b + c.
    $$
    """)
    return


@app.cell
def _(b, f, gradient_descent, gradient_sequence_plot, plt, sigma_A, torch):
    def _():
        gradient_sequence_plot(
            f=f,
            xys=gradient_descent(
                f,
                x_0=torch.tensor([-4.0, 0.0]),
                lr=1e-2,
                n=100,
            ),
            x=torch.linspace(-5.0, 5.0, 100),
            y=torch.linspace(-5.0, 5.0, 100),
        )
        x_min = - (1 / 2) * torch.linalg.inv(sigma_A) @ b

        plt.plot(x_min[0], x_min[1], "k+")
        return plt.gcf()

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Proper quadratic functions
    Note that $A$ is positive definite if and only if $f$ is **proper**, i.e.

    $$
    f(x) \to +\infty \quad \text{ when } \quad \|x\| \to + \infty.
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Hessian Matrix
    Since $\nabla f(x) = \sigma(A) \cdot x + b$, the **Hessian matrix** satisfies:

    $$
    H_f(x) = \nabla^2 f(x) := J_{\nabla f}(x) = \sigma(A).
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gradient Descent Sequence


    /// note | Diagonalization of real symmetric matrices
    Any real-valued symmetric matrix $S \in \mathbb{R}^{d \times d}$ can be diagonalized in an orthonormal basis:

    $$
    S =
        P \cdot
        \Lambda
        \cdot P^{-1}
    $$
    with
    $$
    \Lambda := \text{diag}(\lambda_1, \dots, \lambda_d) =     \begin{bmatrix}
        \lambda_1 & 0         & \cdots    & 0 \\
        0         & \lambda_2 & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & \lambda_d
        \end{bmatrix},
    $$
    and
    $$
    \lambda_{\rm min} := \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_{\rm max} := \lambda_d
    \quad \text{ and } \quad
    P^{-1} = P^T.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We may apply this decomposition to $\sigma(A)$, the Hessian matrix of $f$.
    """)
    return


@app.cell
def _(sigma_A, torch):
    lambdas, P = torch.linalg.eigh(sigma_A)
    return P, lambdas


@app.cell
def _(P, lambdas, torch):
    P @ torch.diag(lambdas) @ P
    return


@app.cell
def _(sigma_A):
    sigma_A
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This decomposition allows us to find an new orthonormal basis in which the axes of the ellipsoid that level sets of $f$ align with the basis axes and the minimum of the function at the origin.

    Indeed, we have

    $$
    \begin{split}
    f(x)
    &= x \cdot (A/2) \cdot x + b \cdot x + c \\
    &=  x \cdot (P \cdot \Lambda/2 \cdot P^T ) + b \cdot x + c \\
    &= (P^T \cdot x) \cdot (\Lambda/2) \cdot (P^T \cdot x) + (P^T \cdot b) \cdot (P^T \cdot x) + c \\
    &= (P^T \cdot x + \Lambda^{-1} \cdot P^T \cdot b) \cdot (\Lambda/2) \cdot (P^T \cdot x + \Lambda^{-1} \cdot P^T \cdot b) + c - (P^T \cdot b)\cdot (\Lambda^{-1}/2) \cdot (P^T \cdot b)
    \end{split}
    $$

    So, in a the new orthogonal frame determined by

    $$
    y := P^T \cdot x + \Lambda^{-1} \cdot P^T \cdot b
    $$

    or equivalently

    $$
    x = P \cdot y - P \cdot \Lambda^{-1} \cdot P^T \cdot b = P \cdot y - \sigma(A)^{-1} \cdot b,
    $$

    we end up with

    $$
    f(x) = y^T \cdot (\Lambda /2) \cdot y +  d
    $$

    where
    $$
    d:=  c - b \cdot \sigma(A)^{-1} \cdot b.
    $$
    """)
    return


@app.cell
def _(lambdas):
    lambdas
    return


@app.cell
def _(P):
    P
    return


@app.cell
def _(P, torch):
    torch.linalg.inv(P)
    return


@app.cell
def _(P):
    P.T
    return


@app.cell
def _(P, b, f, sigma_A, torch):
    def f_diag(y):
        x = torch.einsum("ij,...j->...i", P, y) - torch.einsum("ij,...j->...i", torch.linalg.inv(sigma_A), b)
        return f(x)

    return (f_diag,)


@app.cell
def _(
    P,
    b,
    f_diag,
    gradient_descent,
    gradient_sequence_plot,
    lambdas,
    mo,
    plt,
    torch,
):
    def _():
        gradient_sequence_plot(
            f=f_diag,
            xys=gradient_descent(
                f_diag,
                x_0=(
                    P.T @ torch.tensor([-4.0, 0.0]) - 
                    torch.linalg.inv(torch.diag(lambdas)) @ P.T @ b),
                lr=1e-2,
                n=100,
            ),
            x=torch.linspace(-5.0, 5.0, 100),
            y=torch.linspace(-5.0, 5.0, 100),
        )


        plt.plot(0, 0, "k+")
        return mo.center(plt.gcf())
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gradient Descent Sequence

    Let's explicit the expression of the $n$th term of a gradient descent sequencee when
    $f$ is quadratic. We have:

    $$
    \begin{split}
    x_{n+1}
      & = x_n - \eta \nabla f(x_n) \\
      & = x_n - \eta (\sigma(A) \cdot x_n + b) \\
      & = (I - \eta \sigma(A)) \cdot x_n -\eta b
    \end{split}
    $$

    and thus if we introduce the error $e_n$ between $x_n$ and the minimal argument $x_*$

    $$
    x_* := - \sigma(A)^{-1} \cdot b
    $$

    $$
    e_n := x_{n} - x_*
    $$

    we get

    $$
    \begin{split}
    e_{n+1} &= x_{n+1} + \sigma(A)^{-1} \cdot b \\
                   &= (I - \eta \sigma(A)) \cdot x_n -\eta b +  \sigma(A)^{-1} \cdot b \\
                   &= (I - \eta \sigma(A)) \cdot (x_n +   \sigma(A) ^{-1} \cdot b) +\eta b -\eta b \\
                   &= (I - \eta \sigma(A)) \cdot e_n
    \end{split}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Minimisation argument error

    Let $f(x) = x \cdot (A / 2) \cdot x + b \cdot x + c$ and $x_n$ be the gradient descent sequence associated to the learning parameter $\eta > 0$.
    Then the **minimisation argument error**

    $$
    e_n := x_n - x_*
    $$

    evolves according to

    $$
    e_n = (I - \eta \sigma(A))^n \cdot e_0.
    $$


    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's consider the diagonalisation of $\sigma(A)$

    $$
    \sigma(A) =
        P \cdot
        \begin{bmatrix}
        \lambda_1 & 0         & \cdots    & 0 \\
        0         & \lambda_2 & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & \lambda_d
        \end{bmatrix}
        \cdot P^{-1}
        \quad \text{ with } \quad
        \lambda_{\rm min} :=\lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_{\rm max} = \lambda_d
        \quad \text{ and } \quad
        P^{-1} = P^T.
    $$

    Since $\sigma(A)$ is positive definite, we have

    $$
    0 < \lambda_{\rm min}.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Condition number

    The **condition number** of $\sigma(A)$ is

    $$
    \kappa := \frac{\lambda_{\rm max}}{\lambda_{\rm min}} \geq 1.
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The matrix $1 - \eta \sigma(A)$ satisfies

    $$
    1 - \eta \sigma(A) = P \cdot
        \begin{bmatrix}
        1 - \eta \lambda_1 & 0         & \cdots    & 0 \\
        0         & 1 - \eta \lambda_2 & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & 1 - \eta \lambda_d
        \end{bmatrix}
        \cdot P^{-1}
    $$

    Therefore

    $$
    e_n = P \cdot     \begin{bmatrix}
        (1 - \eta \lambda_1)^n & 0         & \cdots    & 0 \\
        0         & (1 - \eta \lambda_2)^n & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & (1 - \eta \lambda_d)^n
        \end{bmatrix} \cdot P^T \cdot e_0,
    $$

    and the error $e_n$ will converge to $0$ for any $e_0$ if and only if every coefficient $1 - \eta_i \lambda_i$ is in $\left]-1, 1\right[$,
    which holds true if and only if $-1 < 1 - \eta \lambda_{\rm max}$, that is

    $$
    \eta < \frac{2}{\lambda_{\rm max}}.
    $$

    /// note | Edge of Stability
    The gradient descent sequence converges to the minimum as long as
    $$
    \eta < \frac{2}{\lambda_{\rm max}}.
    $$
    Above this threshold, the sequence diverges.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Learning rate: a good choice

    If the learning rate is selected as

    $$
    \eta  = \frac{1}{\lambda_{\max}},
    $$

    (that is, half of the learning rate that is at the edge of stability),
    then for any initial error, the minimisation argument error satisfies

    $$
    \|e_n\| \leq \left(1 - \frac{1}{\kappa} \right)^n \|e_0\|
    \to 0 \quad \text{ when } \quad n \to +\infty.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(b, f, gradient_descent, gradient_sequence_plot, plt, sigma_A, torch):
    def _():
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_A)
        lambda_max = eigenvalues[1]
        lr = 1.0 / lambda_max
        gradient_sequence_plot(
            f=f,
            xys=gradient_descent(
                f,
                x_0=torch.tensor([-4.0, 0.0]),
                lr=lr,
                n=100,
            ),
            x=torch.linspace(-5.0, 5.0, 100),
            y=torch.linspace(-5.0, 5.0, 100),
        )

        x_min = - torch.linalg.inv(sigma_A) @ b

        plt.plot(x_min[0], x_min[1], "k+")
        return plt.gcf()

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// hint | Proof

    The selection $\eta = 1/\lambda_{\rm max}$ obviously satisfies $\eta < 2/\lambda_{\rm max}$ and we get

    $$
    e_n = P \cdot     \begin{bmatrix}
        (1 - \lambda_1 /\lambda_{\rm max})^n & 0         & \cdots    & 0 \\
        0         & (1 - \lambda_2 / \lambda_{\rm max})^n & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & 0
        \end{bmatrix} \cdot P^{-1} \cdot e_0,
    $$

    and thus in the worst case $\|e_n\| \leq (1 - \lambda_1/\lambda_n)^n \|e_0\|$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Learning rate : the best choice

    The best learning rate, which leads to the fastest convergence in the worst-case scenario, is

    $$
    \eta =\frac{2}{\lambda_{\rm max} + \lambda_{\rm min}}
    $$

    For any initial error, it provides

    $$
    \|e_n\| \leq \left(1 - \frac{1+1}{\kappa+1} \right)^n \|e_0\|.
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(b, f, gradient_descent, gradient_sequence_plot, plt, sigma_A, torch):
    def _():
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_A)
        lambda_1, lambda_2 = eigenvalues
        lr = 2 / (lambda_1 + lambda_2)
        gradient_sequence_plot(
            f=f,
            xys=gradient_descent(
                f,
                x_0=torch.tensor([-4.0, 0.0]),
                lr=lr,
                n=100,
            ),
            x=torch.linspace(-5.0, 5.0, 100),
            y=torch.linspace(-5.0, 5.0, 100),
        )

        x_min = - torch.linalg.inv(sigma_A) @ b

        plt.plot(x_min[0], x_min[1], "k+")
        return plt.gcf()

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// warning | Convergence speed and condition number

    Note generally that the higher the condition number $\kappa$, the slower the convergence.

    The best-case scenario is with $\kappa = 1$ which corresponds to $\sigma(A) = \lambda I$ for some $\lambda > 0$. Then, the "good" and the "best" choice of learning rate are identical and lead to a convergence in a single step!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, some simple computations provide $f(x_n) - f(x_*) = e_n \cdot (\sigma(A) /2 ) \cdot e_n$, thus

    $$
    \begin{split}
    f(x_{n+1}) - f(x_*)
      & = e_{n+1} \cdot (\sigma(A)/2) \cdot e_{n+1} \\
      & = e_n \cdot (I - \eta \sigma(A)) \cdot (\sigma(A)/2) \cdot (I - \eta \sigma(A)) e_n \\
      & = e_n \cdot (I - \eta \sigma(A))^2 \cdot (\sigma(A)/2) \cdot e_n
    \end{split}
    $$

    and therefore by induction

    $$
    f(x_n) - f(x_*) = e_0 \cdot (I - \eta \sigma(A))^{2n} \cdot \sigma(A) \cdot e_0.
    $$

    Now for $\eta = 1/\lambda_{\rm max}$, for example, we get

    $$
    \|\sigma(A)\| \leq \lambda_{\rm max}
    \quad \text { and } \quad
    \|I - \eta \sigma(A))\| \leq 1 -\frac{1}{\kappa}
    $$

    and thus

    $$
    \|f(x_n) - f(x_*)\| \leq \left(1-\frac{1}{\kappa}\right)^n \lambda_{\rm \max} \|e_0\|^2
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examples in Higher dimensions

    🚧 *Refactoring needed !*
    """)
    return


@app.cell
def _(torch):
    d = 10
    Sigma = torch.normal(torch.zeros(d, d), torch.ones(d, d))
    return Sigma, d


@app.cell
def _(Sigma):
    A_1 = Sigma.T @ Sigma
    A_1
    return (A_1,)


@app.cell
def _(A_1, torch):
    U, S_1, V = torch.svd(A_1)
    S_1
    return (S_1,)


@app.cell
def _(S_1):
    lambda_1 = S_1[-1].item()
    lambda_1
    return (lambda_1,)


@app.cell
def _(S_1):
    lambda_d = S_1[0].item()
    lambda_d
    return (lambda_d,)


@app.cell
def _(lambda_1, lambda_d):
    kappa = lambda_d / lambda_1
    kappa
    return


@app.cell
def _(A_1):
    def f_3(x):
        return (x @ A_1 @ x).squeeze()

    return (f_3,)


@app.cell
def _(d, f_3, torch):
    lr_7 = 0.001
    x_9 = torch.normal(torch.zeros(d), torch.ones(d))
    x_9.requires_grad = True
    ys_19 = [f_3(x_9).item()]
    print(ys_19[-1])
    optimizer_1 = torch.optim.SGD([x_9], lr=lr_7)
    return optimizer_1, x_9, ys_19


@app.cell
def _(f_3, optimizer_1, x_9, ys_19):
    _n = 10000
    for _i in range(_n):
        _y = f_3(x_9)
        _y.backward()
        ys_19.append(_y.item())
        optimizer_1.step()
        optimizer_1.zero_grad()
    return


@app.cell
def _(plt, ys_19):
    plt.plot(ys_19)
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    return


@app.cell
def _(d, f_3, lambda_d, torch):
    lr_8 = 0.5 / lambda_d
    x_10 = torch.normal(torch.zeros(d), torch.ones(d))
    x_10.requires_grad = True
    ys_20 = [f_3(x_10).item()]
    print(ys_20[-1])
    optimizer_2 = torch.optim.SGD([x_10], lr=lr_8)
    return optimizer_2, x_10, ys_20


@app.cell
def _(f_3, optimizer_2, x_10, ys_20):
    _n = 10000
    for _i in range(_n):
        _y = f_3(x_10)
        _y.backward()
        ys_20.append(_y.item())
        optimizer_2.step()
        optimizer_2.zero_grad()
    return


@app.cell
def _(plt, ys_20):
    plt.plot(ys_20)
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    return


@app.cell
def _(d, f_3, lambda_1, lambda_d, torch):
    lr_9 = 1.0 / (lambda_d + lambda_1)
    x_11 = torch.normal(torch.zeros(d), torch.ones(d))
    x_11.requires_grad = True
    ys_21 = [f_3(x_11).item()]
    print(ys_21[-1])
    optimizer_3 = torch.optim.SGD([x_11], lr=lr_9)
    return optimizer_3, x_11, ys_21


@app.cell
def _(f_3, optimizer_3, x_11, ys_21):
    _n = 10000
    for _i in range(_n):
        _y = f_3(x_11)
        _y.backward()
        ys_21.append(_y.item())
        optimizer_3.step()
        optimizer_3.zero_grad()
    return


@app.cell
def _(plt, ys_21):
    plt.plot(ys_21)
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    return


@app.cell
def _(d, f_3, lambda_d, torch):
    lr_10 = 0.99 / lambda_d
    x_12 = torch.normal(torch.zeros(d), torch.ones(d))
    x_12.requires_grad = True
    ys_22 = [f_3(x_12).item()]
    print(ys_22[-1])
    optimizer_4 = torch.optim.SGD([x_12], lr=lr_10)
    return optimizer_4, x_12, ys_22


@app.cell
def _(f_3, optimizer_4, x_12, ys_22):
    _n = 10000
    for _i in range(_n):
        _y = f_3(x_12)
        _y.backward()
        ys_22.append(_y.item())
        optimizer_4.step()
        optimizer_4.zero_grad()
    return


@app.cell
def _(plt, ys_22):
    plt.plot(ys_22)
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    return


@app.cell
def _(d, f_3, lambda_d, torch):
    lr_11 = 1.01 / lambda_d
    x_13 = torch.normal(torch.zeros(d), torch.ones(d))
    x_13.requires_grad = True
    ys_23 = [f_3(x_13).item()]
    print(ys_23[-1])
    optimizer_5 = torch.optim.SGD([x_13], lr=lr_11)
    return optimizer_5, x_13, ys_23


@app.cell
def _(f_3, optimizer_5, x_13, ys_23):
    _n = 10000
    for _i in range(_n):
        _y = f_3(x_13)
        _y.backward()
        ys_23.append(_y.item())
        optimizer_5.step()
        optimizer_5.zero_grad()
    return


@app.cell
def _(plt, ys_23):
    plt.plot(ys_23)
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    return


if __name__ == "__main__":
    app.run()
