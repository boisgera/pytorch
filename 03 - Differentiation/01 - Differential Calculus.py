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
    # Differential Calculus

    [Sébastien Boisgérault], [ITN], Mines Paris - PSL University

    [ITN]: https://itn.dev
    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Derivative""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Derivative

    Let 

    $$
    x \in \mathbb{R} \mapsto f(x) \in \mathbb{R}^p.
    $$

    The derivative of $f$ at $x \in \mathbb{R}$ is defined as

    $$
    f'(x) := \lim_{\Delta x \to 0} \frac{f(x+\Delta x)-f(x)}{\Delta x} \in \mathbb{R}^p.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// warning

    The concept of derivative works without problems with scalar or vector-valued functions: 

    $$
    f(x) \in \mathbb{R}^p,
    $$

    but it is only relevant for function of a single real variable: 

    $$
    x\in\mathbb{R}.
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    There is an alternate presentation of the concept that will be useful later. Consider
    the function

    $$
    \varepsilon(x, \Delta x) := \left(\frac{f(x+\Delta x)-f(x)}{\Delta x} - f'(x)\right) \frac{\Delta x}{|\Delta x|} 
    $$

    Then $f$ has a derivative at $x$ if and only iff

    $$
    \lim_{\Delta x \to 0} \varepsilon(x, \Delta x) = 0.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Given the definition of $\varepsilon$, that means that $f'(x)$ is uniquely characterized as the vector of $\mathbb{R}^p$ such that

    $$
    f(x + \Delta x) = f(x) + f'(x) \Delta x + \varepsilon(x, \Delta x) |\Delta x|
    $$

    where 

    $$
    \lim_{\Delta x \to 0} \varepsilon(x, \Delta x) = 0.
    $$

    (If no such vector exists, the derivative $f'(x)$ is undefined.)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This is usually written with the Landau notation:

    $$
    f(x + \Delta x) = f(x) + f'(x) \Delta x + o(\Delta x).
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Partial Derivative


    /// note | Partial Derivative

    For functions of several variables:

    $$
    x \in \mathbb{R}^m \to f(x) \in \mathbb{R}^p
    $$

    The partial derivative of $f$ with respect to the $i$th variable $x_i$ at 

    $$
    x=(x_1, \dots, x_m)
    $$

    is defined by:

    $$
    \partial_i f(x) = \frac{\partial f}{\partial x_i}(x) 
    := 
    \lim_{\Delta x_i \to 0} \frac{f(x_1, \dots, x_{i-1}, x_i + \Delta x_i, x_{i+1}, \dots)-f(x)}{\Delta x_i} \in \mathbb{R}^p
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | The partial derivative is a derivative

    Note that if we consider the $i$th partial function $\phi_i$ of $f$ at $x$:

    $$
    \phi_i(t) := f(x_1, \dots, x_{i-1}, t, x_{i+1}, \dots, x_m)
    $$

    then the partial derivative of $f$ at $x$ with respect of the $i$ variable satisfies

    $$
    \partial_i f(x) = \phi_i'(x_i).
    $$
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Jacobian Matrix

    /// note | Jacobian matrix

    Let 

    $$
    x \in \mathbb{R}^m \to f(x) \in \mathbb{R}^p.
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
    \right] \in \mathbb{R}^{p\times m}.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Alternate definition

    If $f$ is decomposed into scalar-valued components

    $$
    f(x) = (f_1(x), \dots, f_i(x), \dots, f_p(x))
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Gradient""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Gradient

    For functions of several variables but a scalar value:

    $$
    f: x \in \mathbb{R}^m \to y \in \mathbb{R}
    $$

    The gradient of $f$ at $x$ is defined as:

    $$
    \nabla f(x) := (\partial_1 f(x), \partial_2 f(x), \dots, \partial_n f(x)) \in \R^n.
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note
    If we identify $\nabla f(x)$ as column vector, then it is equal to the transposed jacobian matrix of $f$ at $x$:

    $$
    \nabla f(x) = J_f(x)^{\top} = 
    \left[ 
    \begin{array}{c}
    \partial_1 f(x) \\
    \partial_2 f(x) \\
    \vdots \\
    \partial_n f(x)
    \end{array}
    \right] \in \R^{n\times 1}.
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | Exercise

    Compute the gradient of:

    $$
    f:(x_1,x_2) \in \R^2 \mapsto (x_2^2 - x_1)^2 + (x_1 - 1)^2 \in \R
    $$ 

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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

    Therefore

    $$
    J_f(x_1, x_2) = 
    \left[ 
      \begin{array}{cc}
      -2(x_2^2 - x_1) + 2 (x_1 - 1) &
      4 (x_2^2 - x_1)x_2
      \end{array}
      \right] \in \R^{1 \times 2}.
    $$

    Its gradient is given by

    $$
    \nabla f(x_1, x_2)
    =
    (-2(x_2^2 - x_1) + 2 (x_1 - 1), 4 (x_2^2 - x_1)x_2) \in \R^2
    $$

    or, represented as a column vector:

    $$
    \nabla f(x_1, x_2) = J_f(x_1, x_2)^{\top} =
    \left[ 
      \begin{array}{c}
      -2(x_2^2 - x_1) + 2 (x_1 - 1) \\
      4 (x_2^2 - x_1)x_2
      \end{array}
      \right] \in \R^{2\times 1}.
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Differential

    /// note | Differentiable function

    Let

    $$
    x \in \mathbb{R}^m \to f(x) \in \mathbb{R}^p.
    $$

    The function $f$ is **differentiable at $x$** if the jacobian matrix is defined at $x$ and

    $$
    f(x + \Delta x) = f(x) + J_f(x) \Delta x + o(\Delta x).
    $$

    In other words, if

    $$
    \frac{f(x + \Delta x) - f(x) - J_f(x) \Delta x}{\|\Delta x\|} \to 0 \; 
    \text{ when } \; \Delta x  \to 0.
    $$

    In this case, the **jacobian-vector product** (jvp) of $f$ at $x$ is the series expansion of order $1$ of $f$ at $x$. 
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Chain Rule

    /// note | Chain Rule
    If 
    $f: \mathbb{R}^p \to \mathbb{R}^{m}$ and
    $g: \mathbb{R}^m \to \mathbb{R}^{r}$ 
    are both differentiable, the composite function $g \circ f$ is differentiable and

    $$
    d(g \circ f)(x) = dg(f(x)) \circ df(x)
    $$

    The jacobian matrix of $g \circ f$ at $x$ satisfies:

    $$
    J_{g \circ f}(x) = J_{g}(f(x)) \cdot J_f(x)
    $$

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// tip | Implementation
    Concretely, if there are three variables $x$, $y$ and $z$, such that

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
    """
    )
    return


if __name__ == "__main__":
    app.run()
