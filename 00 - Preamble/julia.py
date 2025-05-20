import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Julia Sets""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    return plt, torch


@app.cell
def _(torch):
    device = torch.device("cuda") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return (device,)


@app.cell
def _(device, torch):
    # Julia constant
    c = torch.tensor([-0.8, 0.156], dtype=torch.float32, device=device)
    return (c,)


@app.cell
def _():
    # Image dimensions
    width, height = 800, 600

    # Real and imaginary range for the complex plan
    re_min, re_max = -2.0, 2.0
    im_min, im_max = -1.5, 1.5

    # Maximum iterations for escape time
    max_iter = 500
    return height, im_max, im_min, max_iter, re_max, re_min, width


@app.cell
def _(
    c,
    device,
    height,
    im_max,
    im_min,
    max_iter,
    re_max,
    re_min,
    torch,
    width,
):
    def julia():
        re = torch.linspace(re_min, re_max, width, device=device)
        im = torch.linspace(im_min, im_max, height, device=device)
        x, y = torch.meshgrid(re, im, indexing='xy')
        z = torch.stack((x, y), dim=-1)
    
        # Iteration counter
        count = torch.zeros(z.shape[:-1], dtype=torch.int32)
    
        # Start iterations
        z_current = z.clone()
        for i in range(max_iter):
            # Compute z^2
            x, y = z_current[..., 0], z_current[..., 1]
            z_next = torch.stack((
                x**2 - y**2 + c[0],
                2*x*y + c[1]
            ), dim=-1)
    
            # Mask for values not diverged
            mask = (z_current[..., 0]**2 + z_current[..., 1]**2) < 4.0
            count[mask] += 1
    
            z_current[mask] = z_next[mask]
        return count
    return (julia,)


@app.cell(hide_code=True)
def _(im_max, im_min, julia, mo, plt, re_max, re_min):
    plt.figure(figsize=(4, 3))
    plt.imshow(julia(), cmap='inferno', extent=[re_min, re_max, im_min, im_max])
    plt.axis('off')
    plt.tight_layout(pad=0)
    mo.center(plt.gcf())

    return


if __name__ == "__main__":
    app.run()
