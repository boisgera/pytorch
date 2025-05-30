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

    Let $f: \mathbb{R}^d \to \mathbb{R}$ be a differentiable function and let $x_0 \in \mathbb{R}^n$.

    Now let $\eta > 0$; the Taylor expansion of $f$ at $x$ provides:

    \begin{align*}
    f\left(x_0 - \eta \nabla f(x) \right) 
      &= f(x_0) +  J_f(x_0) \cdot  (-\eta \nabla f(x_0)) + o(\eta) \\
      &= f(x_0) - \eta  J_f(x) \cdot  \nabla f(x_0) + o(\eta) \\
      &= f(x_0) - \eta \nabla f(x_0)^T \cdot \nabla f(x_0) + o(\eta) \\
      &= f(x_0) - \eta \|\nabla f(x_0)\|^2 + o(\eta).
    \end{align*}

    So as long as $\nabla f(x_0) \neq 0$, for any $\eta$ small enough, the term $\eta \|\nabla f(x_0)\|^2$ is larger than $o(\eta)$ and if we define

    $$
    x_1 := x_0 - \eta \nabla f(x_0)
    \;\;\; \text{ then } \;\;\;
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


     0. **Learning rate.** Select a small $\eta > 0$, such as $10^{-3}$, 


     1. **Initial guess.** Pick $x_0 \in \mathbb{R}^d$ at random (or guess!).


     2. **Iterations.** Compute repeatdely $x_{n+1} := x_n - \eta \nabla f(x_{n})$,

     3. **Termination.** Stop for example when $f(x_{n+1}) \geq f(x_n)$.


    Then $x_* := x_n$ is your approximation of the minimizer for $f$.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Optimization of Quadratic Functions

    Quadratic functions generalize the 2-order polynomial

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
    ### Why Quadratic Functions?

      - The minimisation of quadratic function solves the (ubiquitous!) linear regression problem: to find the weight matrix $W$ and
        bias $b$ such that $W x + b$ is the best prediction of $y$, solve


        $$
        \min_{W, b} \|W x + b - y \|^2.
        $$


        This criterion which is quadratic w.r.t. $(W, b)$!

      - They are the "best" approximation of a twice continuously differentiable function in a near its minimum $x_*$:

        $$
        f(x) \approx \frac{1}{2} (x - x_*) \nabla^2 f(x_*) (x - x_*) + \nabla f(x_*)^T (x - x_*) + f(x_*).
        $$

      - Many minimisation techniques of quadratic functions generalize to convex functions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | Quadratic Functions in $\mathbb{R}^2$

     1. Pick a random matrices $A \in \mathbb{R}^{2\times2}$, vector $b \in \mathbb{\R}^2$ and scalar $c \in \mathbb{R}$. Define a function `f` that given a torch tensor $x \in \mathbb{R}^2$  returns the scalar

        $$
        f(x) = x^T A x + b^T x + c \in \mathbb{R}.
        $$

     2. Check if that $f$ is proper, that is

        $$
        f(x) \to +\infty \;\;\; \; \text{ when } \;\; \|x\| \to + \infty.
        $$

        If that's not the case, repick $A$ until this property holds!


        Note that every proper quadratic function has a single minimum.


    
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

     3. Now provide a vectorized implementation of $f$, that will associate to any
        tensor $x$ of shape $(i_1, \dots, i_k, 2)$ the corresponding tensor
        $f(x)$ of shape $(i_1, \dots, i_k)$.

     4. Use the [contour](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html) function of matplotlib to represent
        level curves of your functions.

        Adjust the range of the input variables to display a neighborhood of the minimum.
    
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
    _X, _Y = torch.meshgrid(_x, _y)
    _Z = f(torch.stack((_X, _Y), dim=2))
    plt.contour(_X, _Y, _Z, levels=30)
    plt.grid(True)
    plt.colorbar()
    plt.gcf()
    return


@app.cell
def _(f, plt, torch):
    def contour(f, x, y, levels=30):
        _X, _Y = torch.meshgrid(x, y)
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
        Compute "manually" a gradient descent sequence with $n=100$ points. Display this sequence on top of the contour.

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
    def _wrap():
        x = torch.tensor([-4.0, 0.0], requires_grad=True)
        n = 100
        xs = []
        optimizer = torch.optim.SGD(params=[x], lr=1e-2)
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
    _wrap()
    return


@app.cell
def _(f, gradient_sequence_plot, torch):
    def _wrap():
        x = torch.tensor([-4.0, 0.0], requires_grad=True)
        n = 100
        xs = []
        optimizer = torch.optim.SGD(params=[x], lr=1e-2, momentum=0.9)
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
    _wrap()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Theory

    Let $A \in \mathbb{R}^{d \times d}$, $b \in \mathbb{R}^d$, $c \in \mathbb{R}$ and
    $f :\mathbb{R}^d \to \mathbb{R}$ be the function defined by:

    $$
    f(x) := x^T A x + b^T x + c.
    $$

    Then, for any $x \in \mathbb{R}^d$ and $\Delta x \in \mathbb{R}^d$, we have

    $$
    \begin{split}
    df(x) (\Delta x) &= x^T A \Delta x + \Delta x^T A x + b^T \Delta x \\
               &= x^T A \Delta x + x^T A^T \Delta x + b^T \Delta x \\
               &= (x^T A + x^T A^T + b^T) \Delta x \\
               &= (A^Tx + Ax + b)^T \Delta x
    \end{split}
    $$

    Hence,

    $$
    \nabla f(x) = (A + A^T)x + b.
    $$


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let

    $$
    S:= \frac{1}{2}(A + A^T)
    $$

    By construction, the matrix $S$ is **symmetric**: 

    $$
    S^T = S.
    $$

    The computations we have performed previously also yield

    $$
    \nabla f(x) = 2 S x + b.
    $$

    When it is **positive definite**, that is when

    $$
    \forall \, x \in \mathbb{R}^n, \; x^T S x \geq 0 
    \;\;\;
    \text{ and }
    \;\;\;
    x^T S x  = 0 \,\Rightarrow \, x = 0
    $$

    then the function $f(x)$ has a unique (global) minimum $x_*$, solution of $\nabla f(x_*) = 0$, i.e.

    $$
    x_* = - (A + A^T)^{-1} b = - \frac{1}{2} S^{-1} b.
    $$


    and 

    $$
    f(x_*) = x_*^T S x_* + \frac{1}{2} b^T x_* + c = -\frac{1}{4} b^T S^{-1} b + c 
    $$

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    /// note
    Note that 

    $$
    \begin{split}
    f(x) 
      &= \frac{1}{2} x^T A x + \frac{1}{2} x^T A^T x + b^T x + c \\
      & = \frac{1}{2} x^T \left(A + A^T \right) x + b^T x + c \\
      &= x^T S x + b^Tx + c
    \end{split}
    $$

    so we may always enforce the constraint that $A$ is symmetric when we define $f$.
    ///

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note
    Note that $S$ is positive definite if and only if $f$ is **proper**, i.e.

    $$
    f(x) \to +\infty \text{ when } \|x\| \to + \infty.
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
    Since $\nabla f(x) = (A + A^T)x + b = 2 S x + b$, the **Hessian matrix** satisfies:

    $$
    H_f(x) = \nabla^2 f(x) := J_{\nabla f}(x) = (A+ A^T) = 2S.
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    x_{n+1} = x_n - \eta \nabla f(x_n) = x_n - \eta (2S x_n + b) = (I - 2 \eta S) x_n -\eta b
    $$

    $$
    e_n := x_{n} - x_*
    $$

    $$
    \begin{split}
    e_{n+1} - x_*  &= e_{n+1} + \frac{1}{2} S^{-1} b \\
                   &= (I - 2 \eta S) x_n -\eta b + \frac{1}{2} S^{-1} b \\
                   &= (I - 2 \eta S) (x_n +  \frac{1}{2} S^{-1} b) +\eta b -\eta b \\
                   &= (I - 2 \eta S) e_n
    \end{split}
    $$

    The matrix $S$ is real, symmetric and positive definite. Hence, there is an orthogonal matrix $P$ ($P^{-1} = P^T$)
    and $0 < \lambda_1 \leq \dots \leq \lambda_n$ such that.

    $$
    S = P^T \mathrm{diag}(\lambda_1, \dots, \lambda_n)P.
    $$

    The **condition number** of $S$ is

    $$
    \kappa := \frac{\lambda_n}{\lambda_1} \geq 1.
    $$

    The matrix $M := 1 - 2 \eta S$ is also symmetric and

    $$
    M = P^T \mathrm{diag}(1 - 2 \eta \lambda_1, \dots, 1 - 2 \eta \lambda_n) P
    $$

    Since $e_n = P^T M^n P e_0$, the error will converge to $0$ for any $e_0$ iff every coefficient $1 - 2 \eta_i$ is in 
    $\left]-1, 1\right[$, which holds iff $-1 < 1 - 2 \eta \lambda_n$, i.e.

    $$
    \eta < \frac{1}{\lambda_n}.
    $$

    Now, if we pick for example 

    $$
    \eta  = \frac{1}{2\lambda_n}, 
    $$

    we get

    $$
    M = P^T \mathrm{diag}(1 -  \lambda_1/\lambda_n, 1 - \lambda_2/\lambda_n\dots, 0) P
    $$

    and thus

    $$
    \|e_{n+1}\| \leq \left(1 - \frac{1}{\kappa}\right) \|e_n\|  
    $$

    and therefore

    $$
    \|e_n\| \leq \left(1 - \frac{1}{\kappa} \right)^n \|e_0\|.
    $$

    /// note
    The "best" choice, that leads to the fastest convergence is

    $$
    \eta =\frac{1}{\lambda_1 + \lambda_n}
    $$

    which provides

    $$
    \|e_n\| \leq \left(1 - \frac{1+1}{\kappa+1} \right)^n \|e_0\|.
    $$

    ///

    Now, some simple computations provide $f(x) - f(x_*) = e^T S e$, thus

    $$
    f(x_{n+1}) - f(x_*) = e_{n+1}^T S e_{n+1} = e_n^T (I - 2\eta S) S (I - 2\eta S) e_n = e_n^T (I - 2\eta S)^2 S e_n
    $$

    and by induction

    $$
    f(x_n) - f(x_*) = e_0^T (I - 2 \eta S)^{2n}S e_0.
    $$

    Now for $\eta = 1/(2\lambda_n)$, for example, we get

    $$
    \|I - 2 \eta S)\| \leq 1 -\frac{1}{\kappa}
    $$

    $$
    \|S\| \leq \lambda_n
    $$

    and thus

    $$
    \|f(x_n) - f(x_*)\| \leq \left(1-\frac{1}{\kappa}\right)^n \lambda_n \|e_0\|^2
    $$
    """
    )
    return


@app.cell
def _(torch):
    x0_m = 1.0
    x1_m = 2.0
    m = torch.tensor(10.0)
    theta = torch.tensor(torch.pi/4)
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)], 
        [torch.sin(theta), torch.cos(theta)]]
    )
    S = R @ torch.diag(torch.tensor([1.0, 2.0])) @ R.T
    s_00 = S[0, 0].item()
    s_01 = S[0, 1].item()
    s_11 = S[1, 1].item()
    return m, s_00, s_01, s_11, x0_m, x1_m


@app.cell
def _(m, s_00, s_01, s_11, x0_m, x1_m):
    def f_1(x0, x1):
        dx0 = x0 - x0_m
        dx1 = x1 - x1_m
        return 0.5 * (s_00 * dx0 * dx0 + 2.0 * s_01 * dx0 * dx1 + s_11 * dx1 * dx1) + m
    return (f_1,)


@app.cell
def _(torch):
    x_2 = torch.linspace(0.0, 5.0, 100)
    _y = torch.linspace(0.0, 5.0, 100)
    X_1, Y_1 = torch.meshgrid(x_2, _y, indexing='xy')
    (X_1, Y_1)
    return X_1, Y_1


@app.cell
def _(X_1, Y_1, f_1, torch):
    fvmap = torch.vmap(f_1)
    Zf = fvmap(X_1.flatten(), Y_1.flatten())
    Z_1 = Zf.reshape(X_1.shape)
    Z_1
    return (Z_1,)


@app.cell
def _(X_1, Y_1, Z_1, plt, x0_m, x1_m):
    _cs = plt.contour(X_1, Y_1, Z_1, levels=range(0, 30))
    _axes = plt.gca()
    _axes.clabel(_cs)
    plt.plot([x0_m], [x1_m], 'k+')
    plt.colorbar()
    plt.axis('square')
    plt.grid(True)
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Basic Gradient""")
    return


@app.cell
def _(f_1, torch):
    grad_f = torch.func.grad(f_1, (0, 1))
    grad_f(torch.tensor(1.0), torch.tensor(4.0))
    return (grad_f,)


@app.cell
def _(f_1, grad_f, m, torch):
    _N = 100
    lr_3 = 0.1
    xs_14, ys_11 = ([], [])
    x_4 = 4.0
    _y = 1.0
    for _i in range(_N):
        grad_fxy = grad_f(torch.tensor(x_4), torch.tensor(_y))
        dx = grad_fxy[0].item()
        dy = grad_fxy[1].item()
        x_4 = x_4 - lr_3 * dx
        _y = _y - lr_3 * dy
        xs_14.append(x_4)
        ys_11.append(_y)
    es_1 = [f_1(x, _y) - m for x, _y in zip(xs_14, ys_11)]
    return es_1, xs_14, ys_11


@app.cell
def _(grad_f, m, torch):
    def min_grad(f, lr, N=100):
        xs, ys = ([], [])
        x = 4.0
        _y = 1.0
        for _i in range(_N):
            grad_fxy = grad_f(torch.tensor(x), torch.tensor(_y))
            dx = grad_fxy[0].item()
            dy = grad_fxy[1].item()
            x = x - lr * dx
            _y = _y - lr * dy
            xs.append(x)
            ys.append(_y)
        es = [f(x, _y) - m for x, _y in zip(xs, ys)]
        return (xs, ys, es)
    return (min_grad,)


@app.cell
def _(es_1, plt):
    plt.plot(es_1)
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('iteration')
    plt.ylabel('value error')
    None
    return


@app.cell
def _(X_1, Y_1, Z_1, plt, x0_m, x1_m, xs_14, ys_11):
    _cs = plt.contour(X_1, Y_1, Z_1, levels=range(0, 30))
    _axes = plt.gca()
    _axes.clabel(_cs)
    plt.plot([x0_m], [x1_m], 'k+')
    plt.colorbar()
    plt.axis('square')
    plt.grid(True)
    plt.plot(xs_14, ys_11, 'r+-')
    None
    return


@app.cell
def _(f_1, min_grad):
    xs_15, ys_12, es_2 = min_grad(f_1, lr=0.1, N=100)
    return (es_2,)


@app.cell
def _(es_2, plt):
    plt.plot(es_2)
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('iteration')
    plt.ylabel('value error')
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Higher dimensions""")
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
