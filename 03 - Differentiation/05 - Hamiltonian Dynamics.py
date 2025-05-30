import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Hamiltonian Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    H(p, q) = \frac{1}{2} p^T M^{-1}(q) p + V(q)
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    p = M(q) \dot{q}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \dot{q} = \nabla_p H(p, q)
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \dot{p} = -\nabla_q H(p, q) + f
    $$
    """
    )
    return


@app.cell
def _():
    import numpy as np
    from numpy import pi
    return np, pi


@app.cell
def _():
    import torch
    from torch import cos, tensor
    return cos, tensor, torch


@app.cell
def _(np, tensor, torch):
    def F(H):
        def grad_H(q, p):
            q = tensor(q, requires_grad=True, dtype=torch.float64)
            p = tensor(p, requires_grad=True, dtype=torch.float64)
            Hpq = H(q, p)
            Hpq.backward()
            dq = q.grad
            dp = p.grad
            return np.concatenate((dq.numpy(), dp.numpy()), 0)
        def f(t, qp):
            qp = np.array(qp)
            n = len(qp) // 2
            q, p = qp[:n], qp[n:]
            grad_Hqp = grad_H(q, p)
            d_qp = np.concatenate((grad_Hqp[n:], -grad_Hqp[:n]), 0)
            return d_qp
        return f
    return (F,)


@app.cell
def _(cos):
    m = 1
    g = 1

    def K(q, p):
        p = p[0]
        return 0.5 / m * p * p

    def V(q):
        q = q[0]
        return - m * g * cos(q)

    def H(q, p):
        return K(q, p) + V(q)
    return (H,)


@app.cell
def _():
    from scipy.integrate import solve_ivp
    return (solve_ivp,)


@app.cell
def _(F, H, pi, solve_ivp):
    t_span = [0.0, 20.0]
    r = solve_ivp(fun=F(H), y0 = [pi/2, 0.0], t_span=t_span, dense_output=True)
    return r, t_span


@app.cell
def _(np, r, t_span):
    import matplotlib.pyplot as plt
    t = np.linspace(t_span[0], t_span[1], 1000)
    sol = r.sol
    plt.plot(t, sol(t)[0], "C0", label=r"$q$")
    plt.plot(t, sol(t)[1], "C1", label=r"$p$")
    plt.legend()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
