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
    # Images

    [Sébastien Boisgérault], [ITN], Mines Paris - PSL University

    [ITN]: https://itn.dev
    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """
    )
    return


@app.cell
def _():
    import torch
    import torchvision
    return torch, torchvision


@app.cell
def _():
    import pandas as pd
    import PIL
    import matplotlib.pyplot as plt
    import matplotlib.colors 
    import requests
    import sklearn
    import seaborn as sns
    return PIL, matplotlib, pd, requests, sklearn, sns


@app.cell
def _():
    IMAGE_URL = "https://unsplash.com/photos/3k9PGKWt7ik/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8Nnx8cmFpbmJvd3xlbnwwfDB8fHwxNzEyODI5NTAxfDA&force=true&w=640"
    IMAGE_PATH = "cache/colors.jpg"
    return IMAGE_PATH, IMAGE_URL


@app.cell
def _(IMAGE_PATH, IMAGE_URL, requests):
    response = requests.get(IMAGE_URL)
    with open(IMAGE_PATH, mode="bw") as file:
        file.write(response.content)
    return


@app.cell
def _(IMAGE_PATH, PIL):
    image = PIL.Image.open(IMAGE_PATH)
    image
    return (image,)


@app.cell
def _(torchvision):
    image_to_tensor = torchvision.transforms.PILToTensor()
    tensor_to_image = torchvision.transforms.ToPILImage()
    return image_to_tensor, tensor_to_image


@app.cell
def _(image, image_to_tensor):
    t = image_to_tensor(image)
    t
    return (t,)


@app.cell
def _(t):
    t.shape # num_channel, height, width
    return


@app.cell
def _(t, tensor_to_image):
    tensor_to_image(t)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Cropping""")
    return


@app.cell
def _(t):
    t_crop = t[:, 100:-100, 100:-100]
    return (t_crop,)


@app.cell
def _(t_crop, tensor_to_image):
    tensor_to_image(t_crop)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Color Components""")
    return


@app.cell
def _(t):
    R = t[0, :, :]
    R
    return (R,)


@app.cell
def _(R):
    R.shape # height, width
    return


@app.cell
def _(R, tensor_to_image):
    tensor_to_image(R) # Interpreted as greyscale
    return


@app.cell
def _(t, tensor_to_image):
    t_R = t.clone()
    t_R[1, :, :] = 0
    t_R[2, :, :] = 0
    tensor_to_image(t_R)
    return


@app.cell
def _(t, tensor_to_image):
    G = t[1, :, :]
    tensor_to_image(G)
    return (G,)


@app.cell
def _(t, tensor_to_image):
    t_G = t.clone()
    t_G[0, :, :] = 0
    t_G[2, :, :] = 0
    tensor_to_image(t_G)
    return


@app.cell
def _(t, tensor_to_image):
    B = t[2, :, :]
    tensor_to_image(B)
    return (B,)


@app.cell
def _(t, tensor_to_image):
    t_B = t.clone()
    t_B[0, :, :] = 0
    t_B[1, :, :] = 0
    tensor_to_image(t_B)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Images as Floating-Point Data""")
    return


@app.cell
def _(R):
    R.dtype
    return


@app.cell
def _(B, G, R, torch):
    Rf = R.type(torch.float64) / 255  # data should be in the [0, 1] range
    Gf = G.type(torch.float64) / 255
    Bf = B.type(torch.float64) / 255
    return Bf, Gf, Rf


@app.cell
def _(t, torch):
    tf = t.type(torch.float64) / 255
    return (tf,)


@app.cell
def _(tensor_to_image, tf):
    tensor_to_image(tf) # Works too! 👍
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Luminance

    Source: <https://en.wikipedia.org/wiki/Relative_luminance>
    """
    )
    return


@app.cell
def _(Bf, Gf, Rf, tensor_to_image):
    Lf = 0.2126 * Rf + 0.7152 * Gf + 0.0722 * Bf
    tensor_to_image(Lf)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Hue, Saturation, Value (HSV)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Source: <https://en.wikipedia.org/wiki/HSL_and_HSV>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**TODO.** Increase/decrease saturation. Increase/decrase value, Rotate colors?""")
    return


@app.cell
def _(matplotlib, torch):
    def rgb_to_hsv(tf):
        rgb = tf.permute((1, 2, 0)) # matplotlib.colors wants the channels in 3rd dim.
        hsv = torch.tensor(matplotlib.colors.rgb_to_hsv(rgb)).permute((2, 0, 1))
        return hsv # Hue, Saturation and Value in [0, 1]
    return (rgb_to_hsv,)


@app.cell
def _(rgb_to_hsv, tf):
    h, s, v = hsv = rgb_to_hsv(tf)
    return h, hsv, s, v


@app.cell
def _(matplotlib, torch):
    def hsv_to_rgb(hsv):
        hsv = hsv.permute((1, 2, 0)) # matplotlib.colors wants the channels in 3rd dim.
        tf = torch.tensor(matplotlib.colors.hsv_to_rgb(hsv)).permute((2, 0, 1))
        return tf # RGB in [0, 1]
    return (hsv_to_rgb,)


@app.cell
def _(hsv, hsv_to_rgb, tensor_to_image):
    tensor_to_image(hsv_to_rgb(hsv))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Hue Shift""")
    return


@app.cell
def _(h, hsv, hsv_to_rgb, tensor_to_image):
    hue_shift = 180.0 # in degrees
    hue_shifted = (h + (hue_shift / 360.0)) % 1.0
    hsv_hue_shifted = hsv.clone()
    hsv_hue_shifted[0, :, :] = hue_shifted
    tensor_to_image(hsv_to_rgb(hsv_hue_shifted))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""### Saturate & Desaturate""")
    return


@app.cell
def _(hsv, hsv_to_rgb, s, tensor_to_image, torch):
    saturation_boost = 2.0 
    s_saturated = torch.clip(s * saturation_boost, max=1.0)
    hsv_saturated = hsv.clone()
    hsv_saturated[1, :, :] = s_saturated
    tensor_to_image(hsv_to_rgb(hsv_saturated))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Adjust Brightness""")
    return


@app.cell
def _(hsv, hsv_to_rgb, tensor_to_image, torch, v):
    brightness_boost = 5.0 
    v_bright = torch.clip(v * brightness_boost, max=1.0)
    hsv_bright = hsv.clone()
    hsv_bright[2, :, :] = v_bright
    tensor_to_image(hsv_to_rgb(hsv_bright))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Color Statistics""")
    return


@app.cell
def _(tf):
    _, height, width = tf.shape
    return height, width


@app.cell
def _(height, tf, width):
    colors = tf.reshape(3, width*height)
    return (colors,)


@app.cell
def _(colors):
    colors.mean(dim=1)
    return


@app.cell
def _(colors):
    colors.std(dim=1)
    return


@app.cell
def _(colors, pd):
    df = pd.DataFrame(
        {
            "R": colors[0, :],
            "G": colors[1, :],
            "B": colors[2, :],
        }
    )
    df
    return (df,)


@app.cell
def _(df, sns):
    sns.histplot(df, x="R", bins=10, color="red", stat="probability")
    return


@app.cell
def _(df, sns):
    sns.histplot(df, x="G", bins=10, color="green", stat="probability")
    return


@app.cell
def _(df, sns):
    sns.histplot(df, x="B", bins=10, color="blue", stat="probability")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Segmentation

    /// note

    The clustering method that we have used does not work very well in RGB space: the algorithm tends to split dark and bright regions, which is probably not what we want.

    Could you replicate the method to segment the image based on the hue only?

    ///
    """
    )
    return


@app.cell
def _(colors, sklearn):
    kmeans = sklearn.cluster.KMeans(n_clusters=6)
    kmeans.fit(colors.T)
    return (kmeans,)


@app.cell
def _(kmeans):
    kmeans.cluster_centers_
    return


@app.cell
def _(colors, height, kmeans, width):
    category = kmeans.predict(colors.T).reshape((height, width))
    return (category,)


@app.cell
def _(category, tensor_to_image, tf, torch):
    mask_0 = torch.tensor(category == 0)
    tensor_to_image(tf * mask_0)
    return


@app.cell
def _(category, tensor_to_image, tf, torch):
    mask_1 = torch.tensor(category == 1)
    tensor_to_image(tf * mask_1)
    return


@app.cell
def _(category, tensor_to_image, tf, torch):
    mask_2 = torch.tensor(category == 2)
    tensor_to_image(tf * mask_2)
    return


@app.cell
def _(category, tensor_to_image, tf, torch):
    mask_3 = torch.tensor(category == 3)
    tensor_to_image(tf * mask_3)
    return


@app.cell
def _(category, tensor_to_image, tf, torch):
    mask_4 = torch.tensor(category == 4)
    tensor_to_image(tf * mask_4)
    return


@app.cell
def _(category, tensor_to_image, tf, torch):
    mask_5 = torch.tensor(category == 5)
    tensor_to_image(tf * mask_5)
    return


if __name__ == "__main__":
    app.run()
