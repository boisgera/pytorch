import marimo

__generated_with = "0.13.11"
app = marimo.App(layout_file="layouts/01 - Optimization.slides.json")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Optimization


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

    - [ ] Principles of gradient-descent optimization
    - [ ] Manual implementation with function
    - [ ] Pytorch API applied to modules
    - [ ] Example: logistic binary classif with 2 params ?
    - [ ] The "easy" benchmark: quadratic cost function
    - [ ] Theory and practice of GD on Q (Bach, simu)
    - [ ] Learning rates schedulers
    - [ ] SGD with momentum, Adam, RMSProp ?
    - [ ] Stochastic gradient, probabilistic modelling of the pb, batching.

    ///
    """
    )
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
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Gradient Descent Minimization


     0. **Learning rate.** Select a small $\eta > 0$, for example $10^{-3}$, 


     1. **Initial guess.** Pick $x_0 \in \mathbb{R}^d$ at random (or make a guess!).


     2. **Iterations.** Compute repeatedly $x_{n+1} := x_n - \eta \nabla f(x_{n})$,

     3. **Termination.** Stop when $f(x_{n+1}) \geq f(x_n)$ (or your computation budget is over).


    Then your approximation of the minimizer $x_*$ of $f$ is

    $$x_* := x_n.$$ 

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Optimization of Quadratic Functions

    Quadratic functions generalize 2-order polynomials

    $$
    f(x) = a x^2 + b x + c, \; x \in \mathbb{R}
    $$

    to variables in higher dimensions

    $$
    f(x) = x^T A x + b^T x + c, \; x \in \mathbb{R}^d
    $$

    where $A \in \mathbb{R}^{d\times d}$, $b \in \mathbb{R}^d$, $c \in \mathbb{R}^n$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | Why this interest in quadratic functions?

      - The minimisation of quadratic function solves the (ubiquitous!) linear regression problem: to find the weight matrix $A$ and
        bias $b$ such that $A x + b$ is the best prediction of $y$, solve


        $$
        \min_{W, b} \|A x + b - y \|^2.
        $$


        This criterion which is quadratic w.r.t. $(A, b)$[^1]!

      - They approximate locally any (twice continuously differentiable) function at the order 2:

        $$
        f(x) = \frac{1}{2} (x - x_0) \cdot \nabla^2 f(x_*) \cdot (x - x_0) + \nabla f(x_0) \cdot (x - x_0) + f(x_0) + o(\|x-x_0\|)
        $$

      - Many minimisation techniques of quadratic functions generalize to convex functions.

    ///

    [^1]: Suitably "flattened" as a vector of $\mathbb{R}^{d \times d + d}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | Quadratic Functions in $\mathbb{R}^2$

     1. Pick a random matrices $A \in \mathbb{R}^{2\times2}$, vector $b \in \mathbb{\R}^2$ and scalar $c \in \mathbb{R}$. Define in a PyTorch function $f$ that given $x \in \mathbb{R}^2$ returns the scalar

        $$
        f(x) = x \cdot A \cdot x + b \cdot x + c \in \mathbb{R}.
        $$

     2. Check that $f$ is proper, that is

        $$
        f(x) \to +\infty \;\;\; \; \text{ when } \;\; \|x\| \to + \infty.
        $$

        If that's not the case, repick $A$ until this property holds!


        (Every proper quadratic function has a single minimum.)



    ///
    """
    )
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
def _(A, b, c, mo, torch):
    def _f(x):
        return x @ A @ x + b @ x + c
    _x = torch.rand((2,))    
    mo.md(rf"""
    $$
    f({_x[0].item():.4}, {_x[1].item():.4}) = {_f(_x):.4}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | 

     3. Now provide a vectorized PyTorch implementation of $f$, that will associate to any
        tensor $x$ of shape $(i_1, \dots, i_k, 2)$ the corresponding tensor
        $f(x)$ of shape $(i_1, \dots, i_k)$.

     4. Use the [contour](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html) function of matplotlib to represent
        level curves of your functions. Adjust the range of the input variables to display a neighborhood of the minimum.

    ///
    """
    )
    return


@app.cell
def _(A, b, c, torch):
    def f(x):
        return (
            torch.einsum("...i,ij,...j->...", x, A, x) + 
            torch.einsum("i,...i->...", b, x) +
            c
        )

    _x = torch.rand((3, 4, 2))
    f(_x)
    return (f,)


@app.cell
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
    mo.md(
        r"""
    /// tip |

     6. Pick a random initial value of $x_0$ and a value $\eta$ of the learning rate (for example $10^{-3}$).
        Compute a gradient descent sequence with $n=100$ points. Display this sequence on top of the contour.

     7. Change the value of the learning rate and experiment until you have discovered the learning rate at the limit of the convergence.
        Test the gradient sequence with half that value ; experiment with other values until you find (approximately) the learning rate
        that ensures the fastest convergence.

    ///
    """
    )
    return


@app.cell
def _(f, torch):
    def grad_descent(f, x_0, lr=1e-3, n=100):
        xs = []
        x = x_0
        for _ in range(n):
            xs.append(x.clone())
            x.requires_grad = True
            y = f(x)
            y.backward()
            grad_f_x = x.grad
            x = x.detach() - lr * grad_f_x
        return xs

    grad_descent(f, x_0=torch.tensor([-4.0, -4.0]))
    return (grad_descent,)


@app.cell
def _(contour, plt):
    def gradient_sequence_plot(f, xys, x, y):
        fig = contour(f, x, y)
        _xs = [_xy[0] for _xy in xys]
        _ys = [_xy[1] for _xy in xys]
        plt.plot(_xs, _ys, "r-")
        plt.plot(_xs[-1], _ys[-1], "r+")
        return fig
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
            lr=6.39e-1,
            n=100,
        ),
        x=torch.linspace(-5.0, 5.0, 100),
        y=torch.linspace(-5.0, 5.0, 100),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip |

     8. Replicate this computation using the PyTorch [`SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) optimizer class.
    """
    )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Theory

    ### Minimum

    Let $A \in \mathbb{R}^{d \times d}$, $b \in \mathbb{R}^d$, $c \in \mathbb{R}$ and
    $f :\mathbb{R}^d \to \mathbb{R}$ be the function defined by:

    $$
    f(x) := x \cdot A \cdot x + b \cdot x + c.
    $$

    Then, for any $x \in \mathbb{R}^d$ and $\Delta x \in \mathbb{R}^d$, we have

    $$
    \begin{split}
    df(x) (\Delta x) 
      &= \Delta x \cdot  A \cdot x + x \cdot A \cdot \Delta x + b \cdot \Delta x \\
      &= x \cdot A^T \cdot \Delta x + x \cdot A \cdot \Delta x + b \cdot \Delta x \\
      &= (x\cdot A^T + x\cdot A + b) \cdot \Delta x \\
      &= (A \cdot x + A^T\cdot x + b) \cdot \Delta x \\
      &= ((A + A^T) \cdot x + b) \cdot \Delta x \\
    \end{split}
    $$

    Hence,

    $$
    \nabla f(x) = (A + A^T) \cdot x + b.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(A):
    sigma_A = (A + A.T) / 2
    sigma_A
    return (sigma_A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Symmetry Assumption
    Note that since

    $$
    \begin{split}
    f(x) 
      &= x^T A x + b^T x + c \\
      &= \frac{1}{2} x^T A x + \frac{1}{2} x^T A^T x + b^T x + c \\
      & = \frac{1}{2} x^T \left(A + A^T \right) x + b^T x + c \\
      &= x^T \sigma(A) x + b^Tx + c
    \end{split}
    $$

    we may always replace $A$ with its symmetrization (or assume that $A$ was symmetric in the first place!).
    ///

    The computations we have performed previously yield

    $$
    \nabla f(x) = 2 \sigma(A) x + b.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Existence, uniqueness and value of the minimum
    If $A$ is positive definite, then the function $f(x)$ has a unique global minimum.

    The minimal argument $x_*$ is the (unique) solution of 

    $$
    \nabla f(x_*) = 0
    $$

    which is

    $$
    x_* = - \frac{1}{2} \sigma(A)^{-1} \cdot b.
    $$

    and consequently

    $$
    f(x_*) = -\frac{1}{4} b \cdot \sigma(A)^{-1} \cdot b + c.
    $$
    """
    )
    return


@app.cell
def _(S, b, f, grad_descent, gradient_sequence_plot, plt, torch):
    def _():
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
        x_min = - (1 / 2) * torch.linalg.inv(S) @ b

        plt.plot(x_min[0], x_min[1], "k+")
        return plt.gcf()

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Proper quadratic functions
    Note that $A$ is positive definite if and only if $f$ is **proper**, i.e.

    $$
    f(x) \to +\infty \quad \text{ when } \quad \|x\| \to + \infty.
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Hessian Matrix
    Since $\nabla f(x) = 2 \sigma(A) \cdot x + b$, the **Hessian matrix** satisfies:

    $$
    H_f(x) = \nabla^2 f(x) := J_{\nabla f}(x) = 2\sigma(A).
    $$

    ///
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Gradient Descent Sequence


    /// note | Diagonalization of real symmetric matrices
    Any real-valued symmetric matrix $S \in \mathbb{R}^{d \times d}$ can be diagonalized in an orthonormal basis:

    $$
    S = 
        Q \cdot  
        \begin{bmatrix}
        \lambda_1 & 0         & \cdots    & 0 \\
        0         & \lambda_2 & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & \lambda_d
        \end{bmatrix}
        \cdot Q^T 
        \quad \text{ with } \quad
        \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_n
        \quad \text{ and } \quad    
        Q^T = Q^{-1}.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Applied to $S = \sigma(A)$, this result allows us to find an orthonormal basis in which the axes of the ellipsoid that level sets of $f$ align with the basis axes.""")
    return


@app.cell
def _(sigma_A, torch):
    lambdas, Q = torch.linalg.eigh(sigma_A)
    return Q, lambdas


@app.cell
def _(lambdas):
    lambdas
    return


@app.cell
def _(Q):
    Q
    return


@app.cell
def _(Q, f, torch):
    def f_diag(y):
        return f(torch.einsum("ij,...j->...i", Q, y))
    return (f_diag,)


@app.cell
def _(Q, S, b, f_diag, grad_descent, gradient_sequence_plot, plt, torch):
    def _():
        gradient_sequence_plot(
            f=f_diag,
            xys=grad_descent(
                f_diag,
                x_0=(Q @ torch.tensor([-4.0, 0.0])),
                lr=1e-2,
                n=100,
            ),
            x=torch.linspace(-5.0, 5.0, 100),
            y=torch.linspace(-5.0, 5.0, 100),
        )

        x_min = - (1 / 2) * torch.linalg.inv(S) @ b
        y_min = Q @ x_min 

        plt.plot(y_min[0], y_min[1], "k+")
        return plt.gcf()
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Gradient Descent Sequence

    Let's explicit the expression of the $n$th term of a gradient descent sequencee when
    $f$ is quadratic. We have:

    $$
    \begin{split}
    x_{n+1} 
      & = x_n - \eta \nabla f(x_n) \\
      & = x_n - \eta (2\sigma(A) \cdot x_n + b) \\
      & = (I - 2 \eta \sigma(A)) \cdot x_n -\eta b
    \end{split}
    $$

    and thus if we introduce the error between $x_n$ and the minimal argument $x_*$

    $$
    e_n := x_{n} - x_*
    $$

    we get

    $$
    \begin{split}
    e_{n+1} &= e_{n+1} + \frac{1}{2} \sigma(A)^{-1} \cdot b \\
                   &= (I - 2 \eta \sigma(A)) x_n -\eta b + \frac{1}{2} \sigma(A)^{-1} \cdot b \\
                   &= (I - 2 \eta \sigma(A)) \cdot (x_n +  \frac{1}{2} \sigma(A) ^{-1} \cdot b) +\eta b -\eta b \\
                   &= (I - 2 \eta \sigma(A)) \cdot e_n
    \end{split}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Minimisation argument error

    Let $f(x) = x \cdot A \cdot x + b \cdot x + c$ and $x_n$ be the gradient descent sequence associated to the learning parameter $\eta > 0$.
    Then the **minimisation argument error**

    $$
    e_n := x_n - x_*
    $$

    evolves according to 

    $$
    e_n = (I - 2 \eta \sigma(A))^n \cdot e_0.
    $$


    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's consider the diagonalisation of $\sigma(A)$

    $$
    \sigma(A) = 
        Q \cdot  
        \begin{bmatrix}
        \lambda_1 & 0         & \cdots    & 0 \\
        0         & \lambda_2 & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & \lambda_d
        \end{bmatrix}
        \cdot Q^T 
        \quad \text{ with } \quad
        \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_n
        \quad \text{ and } \quad    
        Q^T = Q^{-1}.
    $$

    Since $\sigma(A)$ is positive definite, we have 

    $$
    0 < \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_n.
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    /// note | Condition number

    The **condition number** of $\sigma(A)$ is 

    $$
    \kappa := \frac{\lambda_n}{\lambda_1} \geq 1.
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The matrix $1 - 2 \eta \sigma(A)$ satisfies

    $$
    1 - 2 \eta \sigma(A) = Q \cdot 
        \begin{bmatrix}
        1 - 2 \eta \lambda_1 & 0         & \cdots    & 0 \\
        0         & 1 - 2 \eta \lambda_2 & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & 1 - 2 \eta \lambda_d
        \end{bmatrix}
        \cdot Q^T
    $$

    Therefore

    $$
    e_n = Q \cdot     \begin{bmatrix}
        (1 - 2 \eta \lambda_1)^n & 0         & \cdots    & 0 \\
        0         & (1 - 2 \eta \lambda_2)^n & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & (1 - 2 \eta \lambda_d)^n
        \end{bmatrix} \cdot Q^T \cdot e_0,
    $$

    and the error $e_n$ will converge to $0$ for any $e_0$ if and only if every coefficient $1 - 2 \eta_i$ is in $\left]-1, 1\right[$, 
    which holds true if and only if $-1 < 1 - 2 \eta \lambda_d$, that is

    $$
    \eta < \frac{1}{\lambda_d}.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Learning rate: a good choice

    If the learning rate is selected as

    $$
    \eta  = \frac{1}{2\lambda_d}, 
    $$

    then for any initial error, the minimisation argument error satisfies

    $$
    \|e_n\| \leq \left(1 - \frac{1}{\kappa} \right)^n \|e_0\|
    \to 0 \quad \text{ when } \quad n \to +\infty.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(b, f, grad_descent, gradient_sequence_plot, plt, sigma_A, torch):
    def _():
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_A)
        lambda_2 = eigenvalues[1]
        lr = 1 / (2 * lambda_2)
        gradient_sequence_plot(
            f=f,
            xys=grad_descent(
                f,
                x_0=torch.tensor([-4.0, 0.0]),
                lr=lr,
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
    mo.md(
        r"""
    /// hint | Proof

    The selection $\eta = 1/2\lambda_d$ obviously satisfies $\eta < 1/\lambda_d$ and
    we get

    $$
    e_n = Q \cdot     \begin{bmatrix}
        (1 - \lambda_1 /\lambda_d)^n & 0         & \cdots    & 0 \\
        0         & (1 - \lambda_2 / \lambda_d)^n & \cdots    & 0 \\
        \vdots    & \vdots    & \ddots    & \vdots \\
        0         & 0         & \cdots    & 0
        \end{bmatrix} \cdot Q^T \cdot e_0,
    $$

    and thus in the worst case $\|e_n\| \leq (1 - \lambda_1/\lambda_n)^n \|e_0\|$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Learning rate : the best choice

    The best learning rate, which leads to the fastest convergence in the worst-case scenario, is

    $$
    \eta =\frac{1}{\lambda_1 + \lambda_n}
    $$

    For any initial error, it provides

    $$
    \|e_n\| \leq \left(1 - \frac{1+1}{\kappa+1} \right)^n \|e_0\|.
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(b, f, grad_descent, gradient_sequence_plot, plt, sigma_A, torch):
    def _():
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_A)
        lambda_1, lambda_2 = eigenvalues
        lr = 1 / (lambda_1 + lambda_2)
        gradient_sequence_plot(
            f=f,
            xys=grad_descent(
                f,
                x_0=torch.tensor([-4.0, 0.0]),
                lr=lr,
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
    mo.md(
        r"""
    /// warning | Convergence speed and condition number

    Note generally that the higher the condition number $\kappa$, the slower the convergence. 

    The best-case scenario is with $\kappa = 1$ which corresponds to $\sigma(A) = \lambda I$ for some $\lambda > 0$; then, the "good" and the "best" choice of learning rate are identical and lead to a convergence in a single step!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, some simple computations provide $f(x_n) - f(x_*) = e_n \cdot S \cdot e_n$, thus

    $$
    \begin{split}
    f(x_{n+1}) - f(x_*) 
      & = e_{n+1} \cdot \sigma(A) \cdot e_{n+1} \\
      & = e_n \cdot (I - 2\eta \sigma(A)) \cdot \sigma(A) \cdot (I - 2\eta \sigma(A)) e_n \\
      & = e_n \cdot (I - 2\eta \sigma(A))^2 \cdot \sigma(A) \cdot e_n
    \end{split}
    $$

    and therefore by induction

    $$
    f(x_n) - f(x_*) = e_0^T (I - 2 \eta \sigma(A))^{2n} \sigma(A) e_0.
    $$

    Now for $\eta = 1/(2\lambda_n)$, for example, we get

    $$
    \|\sigma(A)\| \leq \lambda_n
    \quad \text { and } \quad
    \|I - 2 \eta \sigma(A))\| \leq 1 -\frac{1}{\kappa}
    $$

    and thus

    $$
    \|f(x_n) - f(x_*)\| \leq \left(1-\frac{1}{\kappa}\right)^n \lambda_n \|e_0\|^2
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Examples in Higher dimensions

    🚧 *Refactoring needed !*
    """
    )
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
