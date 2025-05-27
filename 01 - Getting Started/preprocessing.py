import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageEnhance
    import scipy.ndimage
    return PIL, np, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Preprocessing""")
    return


@app.cell
def _():
    image_filename = "images/columbia.jpg" # "kiprun.jpg" "columbia.jpg"
    return (image_filename,)


@app.cell
def _(PIL, image_filename):
    image_thumbnail = PIL.Image.open(image_filename)
    image_thumbnail.thumbnail((200, 200))
    image_thumbnail
    return


@app.cell
def _(PIL, image_filename):
    image = PIL.Image.open(image_filename) # Too large to be displayed!
    aspect_ratio = image.width / image.height
    aspect_ratio # should be similar to 4/3
    error = abs((aspect_ratio - 4/3) / (4/3))
    assert error < 0.01
    return error, image


@app.cell(hide_code=True)
def _(error, mo):
    mo.md(rf"""The difference between the aspect ratio of the original image and 4/3 is approximately {100.0 * error:.1}%.""")
    return


@app.cell
def _(image):
    image_800x600 = image.resize((800, 600))
    image_800x600
    return (image_800x600,)


@app.cell
def _(PIL, image_800x600):
    mask_0 = PIL.Image.new("L", (800, 600))

    def is_green(r, g, b, threshold=40, min_green=60):
        return (g - r > threshold) and (g - b > threshold) and g > min_green

    # Loop through pixels and set mask
    for y in range(mask_0.height):
        for x in range(mask_0.width):
            r, g, b = image_800x600.getpixel((x, y))
            if is_green(r, g, b):
                mask_0.putpixel((x, y), 0) # black where green
            else:
                mask_0.putpixel((x, y), 255)

    mask_0
    return (mask_0,)


@app.cell
def _(PIL, mask_0):
    mask_1 = mask_0.copy()

    PIL.ImageDraw.floodfill(mask_1, (0, 0), 0)
    mask_1
    return (mask_1,)


@app.cell
def _(PIL, mask_1, np, scipy):
    mask_2 = np.array(mask_1)

    binary = (mask_2 > 128).astype(np.uint8)

    # Apply erosion followed by dilation
    kernel = np.ones((2, 2), dtype=np.uint8)
    eroded = scipy.ndimage.binary_erosion(binary, structure=kernel, iterations=1)
    cleaned = scipy.ndimage.binary_dilation(eroded, structure=kernel, iterations=1)

    # Convert boolean result back to 0-255 image
    cleaned_mask = (cleaned * 255).astype(np.uint8)
    mask_3 = PIL.Image.fromarray(cleaned_mask, "L")
    mask_3
    return (mask_3,)


@app.cell
def _(PIL, image_800x600):
    image_bright_800x600 = PIL.ImageEnhance.Brightness(image_800x600).enhance(2.0)
    image_bright_800x600
    return (image_bright_800x600,)


@app.cell
def _(PIL, image_bright_800x600, mask_3):
    black_800x600 = PIL.Image.new("L", (800,600))
    image_masked = PIL.Image.composite(image_bright_800x600, black_800x600, mask_3)
    image_masked
    return (image_masked,)


@app.cell
def _(image_masked):
    image_cropped = image_masked.crop(image_masked.getbbox())
    image_cropped
    return (image_cropped,)


@app.cell
def _(image_cropped):
    image_greyscale = image_cropped.convert("L")
    image_greyscale
    return


@app.cell
def _(PIL, image_cropped):
    image_square = PIL.Image.new("L", (image_cropped.width, image_cropped.width))
    image_square.paste(image_cropped, (0, round((600-image_cropped.height)/2)))
    image_square
    return (image_square,)


@app.cell
def _(image_square):
    image_28x28 = image_square.resize((28,28))
    image_28x28
    return (image_28x28,)


@app.cell
def _(image_28x28, image_filename):
    image_28x28.save(image_filename.replace(".jpg", "-28x28.png"))
    return


if __name__ == "__main__":
    app.run()
