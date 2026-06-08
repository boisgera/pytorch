import marimo

__generated_with = "0.23.5"
app = marimo.App(layout_file="layouts/01 - Differential Calculus.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Differential Calculus

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derivative
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Derivative of Scalar-Valued Functions

    Let

    $$
    x \in \mathbb{R} \mapsto f(x) \in \mathbb{R}.
    $$

    The derivative of $f$ at $x \in \mathbb{R}$ is defined as

    $$
    f'(x) := \lim_{\Delta x \to 0} \frac{f(x+\Delta x)-f(x)}{\Delta x} \in \mathbb{R}
    $$

    whenever the limit exists.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Derivative of Vector-Valued Functions

    Let

    $$
    x \in \mathbb{R} \mapsto f(x) \in \mathbb{R}^m.
    $$

    The derivative of $f$ at $x \in \mathbb{R}$ is defined as

    $$
    f'(x) := \lim_{\Delta x \to 0} \frac{f(x+\Delta x)-f(x)}{\Delta x} \in \mathbb{R}^m
    $$

    whenever the limit exists.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// warning

    The concept of derivative can also be defined for matrix-valued function or even for generally for arbitrary tensor-valued functions. However, it is only applicable for functions of a single scalar variable.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Derivatives of Vector-Valued Functions

    The derivative of the function

    $$
    x \in \mathbb{R} \mapsto f(x) \in \mathbb{R}^m
    $$

    exists at $x\in \mathbb{R}$ if and only if for every $i \in {1, \dots, n}$, the derivative of the component function


    $$
    [f(\bullet )]_i \in \mathbb{R}: x \in \mathbb{R} \mapsto [f(x)]_i \in \mathbb{R}
    $$

    exists. When this is the case, then for every $i \in {1, \dots, n}$,

    $$
    [f'(x)]_i = \left([f(\bullet)]_{i}\right)'(x).
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There is an alternate presentation of the concept that will be useful later. If $f$ has a derivative at $x \in \mathbb{R}$, then the function

    $$
    \varepsilon_x(\Delta x) := \left(\frac{f(x+\Delta x)-f(x)}{\Delta x} - f'(x)\right) \frac{\Delta x}{|\Delta x|}
    $$

    satisfies

    $$
    \lim_{\Delta x \to 0} \varepsilon_x(\Delta x) = 0.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given the definition of $\varepsilon$, that means that $f'(x)$ is uniquely characterized as the scalar/vector such that

    $$
    f(x + \Delta x) = f(x) + f'(x) \Delta x + \varepsilon_x(\Delta x) |\Delta x|
    $$

    where

    $$
    \lim_{\Delta x \to 0} \varepsilon_x(\Delta x) = 0.
    $$

    (If no such vector exists, the derivative $f'(x)$ is undefined.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is usually written with the [Landau] notation:

    $$
    f(x + \Delta x) = f(x) + f'(x) \Delta x + o(\Delta x).
    $$

    [Landau]: https://en.wikipedia.org/wiki/Edmund_Landau
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partial Derivative


    /// note | Partial Derivative

    For functions of several variables:

    $$
    x \in \mathbb{R}^n \to f(x) \in \mathbb{R} \; (\text{or } \mathbb{R}^{m})
    $$

    The partial derivative of $f$ with respect to the $i$th variable $x_i$ at

    $$
    x=(x_1, \dots, x_n)
    $$

    is defined by:

    $$
    \partial_i f(x) = \frac{\partial f}{\partial x_i}(x)
    :=
    \lim_{\Delta x_i \to 0} \frac{f(x_1, \dots, x_{i-1}, x_i + \Delta x_i, x_{i+1}, \dots)-f(x)}{\Delta x_i} \in \mathbb{R}  \; (\text{or } \mathbb{R}^{m}).
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | The partial derivative is a derivative

    Note that if we consider the $i$th partial function $\phi_i$ of $f$ at $x$:

    $$
    \phi_i(t) := f(x_1, \dots, x_{i-1}, t, x_{i+1}, \dots, x_n)
    $$

    then the partial derivative of $f$ at $x$ with respect of the $i$ variable satisfies

    $$
    \partial_i f(x) = \phi_i'(x_i).
    $$
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scalar-Valued Functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Gradient

    For scalar-valued functions of several variables:

    $$
    f: x \in \mathbb{R}^n \to y \in \mathbb{R},
    $$

    the gradient of $f$ at $x$ is defined as:

    $$
    \nabla f(x) := (\partial_1 f(x), \partial_2 f(x), \dots, \partial_n f(x)) \in \R^n
    $$

    whenever it exists.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Exercise

    Compute the gradient of:

    $$
    f:(x_1,x_2) \in \R^2 \mapsto (x_2^2 - x_1)^2 + (x_1 - 1)^2 \in \R
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// hint | Solution

    Both partial functions of

    $$
    f:(x_1,x_2) \in \R^2 \mapsto (x_2^2 - x_1)^2 + (x_1 - 1)^2 \in \R
    $$

    have a derivative ;
    they satisfy

    $$
    \partial_1 f(x_1, x_2) = -2(x_2^2 - x_1) + 2 (x_1 - 1)
    $$

    and

    $$
    \partial_2 f(x_1, x_2) = 4 (x_2^2 - x_1)x_2.
    $$

    Therefore its gradient is given by

    $$
    \nabla f(x_1, x_2)
    =
    (-2(x_2^2 - x_1) + 2 (x_1 - 1), 4 (x_2^2 - x_1)x_2) \in \R^2.
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vector-Valued Functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Jacobian matrix

    Let

    $$
    x \in \mathbb{R}^n \to f(x) \in \mathbb{R}^m.
    $$

    The **jacobian matrix** of $f$ at $x$ is defined as:

    $$
    J_f(x) :=
    \left[
    \begin{array}{cccc}
    \vert & \vert & \cdots & \vert \\
    \partial_1 f (x) & \partial_2 f (x) & \cdots & \partial_n f (x) \\
    \vert & \vert & \cdots & \vert \\
    \end{array}
    \right] \in \mathbb{R}^{m\times n}.
    $$
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Alternate definition

    If $f$ is decomposed into scalar-valued components

    $$
    f(x) = (f_1(x), \dots, f_i(x), \dots, f_n(x))
    $$

    then we have

    $$
    J_f(x) :=
    \left[
    \begin{array}{cccc}
    \partial_1 f_1 (x) & \partial_2 f_1 (x) & \cdots & \partial_n f_1 (x) \\
    \partial_1 f_2 (x) & \partial_2 f_2 (x) & \cdots & \partial_n f_2 (x) \\
    \vdots & \vdots & \vdots & \vdots \\
    \partial_1 f_m (x) & \partial_2 f_m (x) & \cdots & \partial_n f_m (x) \\
    \end{array}
    \right]
    $$

    or in compact form:

    $$
    J_f(x) := \left[\partial_{j} f_i(x)\right]_{ij}.
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note
    In the special case where $f(x)$ is a vector of dimension 1

    $$
    x \in \mathbb{R}^n \to f(x) \in \mathbb{R}^1
    $$

    we can identify $f(x)$ as its only scalar component, compute its gradient $\nabla f(x) \in \mathbb{R}$ and finally identify this gradient with a row vector of $\mathbb{R}^{1 \times n}$. If we do this, then we have:

    $$
    J_f(x) = \nabla f(x) =
    \left[
    \begin{array}{c}
    \partial_1 f(x) &
    \partial_2 f(x) &
    \cdots &
    \partial_n f(x)
    \end{array}
    \right] \in \R^{1\times n}.
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Exercise
    Show that for every $x=(x_1, x_2)$ de $\R^2$, the jacobian matrix of the function

    $$
    f:(x_1, x_2) \in \R^2 \mapsto (-2(x_2^2 - x_1) + 2 (x_1 - 1), 4 (x_2^2 - x_1)x_2) \in \R^2.
    $$

    is defined and satisfies

    $$
    J_f(x_1, x_2) =
    \left[
      \begin{array}{cc}
      4 & -4x_2 \\
      -4x_2 & 12 x_2^2
      \end{array}
      \right]\in \R^{2 \times 2}.
    $$
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// hint | Solution
    We have $f(x) = (f_1(x), f_2(x))$ with

    $$
    f_1(x) := -2(x_2^2 - x_1) + 2 (x_1 - 1)
    \;\;\;
    \text{ and }
    \;\;\;
    f_2(x) = 4 (x_2^2 - x_1)x_2.
    $$

    Both functions have partial derivatives with respect to their two arguments:

    $$
    \partial_1 f_1(x) = 4,
    \; \; \;
    \partial_2 f_1(x) = -4 x_2,
    $$

    $$
    \partial_1 f_2(x) = -4x_2,
    \; \; \;
    \partial_2 f_2(x) = 12 x_2^2
    $$

    Finally,

    $$
    J_f(x_1, x_2) =
    \left[
      \begin{array}{cc}
      4 & -4x_2 \\
      -4x_2 & 12 x_2^2
      \end{array}
      \right]\in \R^{2 \times 2}.
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differential
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Differentiable Scalar-Valued Function

    Let

    $$
    x \in \mathbb{R}^n \to f(x) \in \mathbb{R}.
    $$

    The function $f$ is **differentiable at $x$** if its gradient is defined at $x$ and

    $$
    f(x + \Delta x) = f(x) + \nabla f(x) \cdot \Delta x + o(\Delta x),
    $$

    that is, if

    $$
    \frac{f(x + \Delta x) - f(x) - \nabla f(x) \cdot \Delta x}{\|\Delta x\|} \to 0 \;
    \text{ when } \; \Delta x  \to 0.
    $$

    The **differential** $df(x)$ at $x$ is associates to a vector $\Delta x$ the gradient-vector contraction:

    $$
    \Delta x \in \mathbb{R}^m \to df(x)(\Delta x) := \nabla f(x) \cdot \Delta x\in \mathbb{R}.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Differentiable Vector-Valued Function

    Let

    $$
    x \in \mathbb{R}^n \to f(x) \in \mathbb{R}^m.
    $$

    The function $f$ is **differentiable at $x$** if the jacobian matrix is defined at $x$ and

    $$
    f(x + \Delta x) = f(x) + J_f(x) \cdot\Delta x + o(\Delta x),
    $$

    that is, if

    $$
    \frac{f(x + \Delta x) - f(x) - J_f(x) \cdot \Delta x}{\|\Delta x\|} \to 0 \;
    \text{ when } \; \Delta x  \to 0.
    $$


    The **differential** $df(x)$ at $x$ associates to a vector $\Delta x$ the jacobian-vector contraction:

    $$
    \Delta x \in \mathbb{R}^m \to df(x)(\Delta x) := J_f(x) \cdot \Delta x\in \mathbb{R}^m.
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Differential

    If $f$ is differentiable at $x$, its differential $df(x)$ at $x$ is the function

    $$
    \Delta x \in \mathbb{R}^m \to df(x)(\Delta x) \in \mathbb{R}^p
    $$

    that associates to a vector $\Delta x$ the corresponding jacobian-vector product:

    $$
    df(x)(\Delta x) := J_f(x) \Delta x
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chain Rule

    /// note | Chain Rule
    If
    $f: \mathbb{R}^p \to \mathbb{R}^{n}$ and
    $g: \mathbb{R}^n \to \mathbb{R}^{m}$
    are both differentiable, the composite function $g \circ f$ is differentiable and

    $$
    d(g \circ f)(x) = dg(f(x)) \circ df(x)
    $$

    The jacobian matrix of $g \circ f$ at $x$ satisfies:

    $$
    J_{g \circ f}(x) = J_{g}(f(x)) \cdot J_f(x)
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Implementation
    Concretely, if there are three vector variables $x$, $y$ and $z$, such that

    $$
    y = f(x) \; \text{ and } \; z= g(y),
    $$

    to compute the first-order variation of $z$ with respect to $x$, one can compute

    $$
    \Delta y = J_f(x) \Delta x
    \;
    \text{ and then }
    \;
    \Delta z  = J_g(y) \Delta y
    $$

    or, using indices and partial derivatives

    $$
    \frac{\partial z_i}{\partial x_j}(x) = \sum_k \frac{\partial z_i}{\partial y_k}(y) \frac{\partial y_k}{\partial x_i}(x).
    $$

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hessian

    /// note | Hessian
    Assume that
    $f: \mathbb{R}^n \mapsto \mathbb{R}$ is **twice differentiable**: $f$ and $\nabla f$ are differentiable.

    The **Hessian** of $f$ at $x$, denoted $H_f(x)$ or $\nabla^2 f(x)$ is defined as

    $$
    H_f(x) := J_{\nabla f}(x)
    $$

    or equivalently

    $$
    [H_{\nabla f}(x)]_{ij} = \frac{\partial^2 f(x)}{\partial x_i \partial x_j}
    $$

    is a symmetric matrix.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sharpness

    /// note | Sharpness

    If $f$ is twice differentiable, its hessian at $x$ can be factored out as

    $$
    H_f(x) = P \times \mathrm{diag}(\lambda_1, \lambda_2, \dots, \lambda_n) \times P^T.
    $$

    where $P$ is orthogonal

    $$
    P \times P^T = I
    $$

    and the **eigenvalues** $\lambda_i$ of the hessian are real and sorted:

    $$
    \lambda_1 \ge \lambda_2 \geq ... \geq \lambda_n.
    $$

    The largest eigenvalue $\lambda_1$ is the **sharpness** of $f$ at $x$.

    ///
    """)
    return


if __name__ == "__main__":
    app.run()
