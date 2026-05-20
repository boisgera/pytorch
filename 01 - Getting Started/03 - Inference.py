import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inference

    [Sébastien Boisgérault], Mines Paris - PSL University

    Source: [Pytorch tutorials](https://docs.pytorch.org/tutorials/)


    [Sébastien Boisgérault]: mailto://Sebastien.Boisgerault@minesparis.psl.eu
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Learning Objectives
    - [ ] Load a pre-trained neural network model,
    - [ ] Understand its output,
    - [ ] Use it for classification,
    - [ ] Use it in batched mode,
    - [ ] Evaluate the confidence in its answers.

    ///
    """)
    return


@app.cell
def _():
    import torch
    import torchvision

    return torch, torchvision


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import PIL.Image
    import seaborn as sns; sns.set_theme()
    return PIL, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We download the subset of the FashionMNIST dataset that has *not* been used to train our model and put it in the `cache` directory.
    """)
    return


@app.cell
def _(torchvision):
    dataset = torchvision.datasets.FashionMNIST(root="cache", download=True, train=False)
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    index = 0 # Select your item
    assert 0 <= index < len(dataset)
    return (index,)


@app.cell
def _(dataset, index):
    datum = dataset[index]
    image, cls = datum
    return cls, image


@app.cell
def _(image):
    image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Architecture
    """)
    return


@app.cell
def _(torch):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28*28, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )

    model
    return (model,)


@app.cell
def _(torch):
    state_dict = torch.load("models/model.pth")
    state_dict
    return (state_dict,)


@app.cell
def _(model, state_dict):
    model.load_state_dict(state_dict)
    model
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Transformation
    """)
    return


@app.cell
def _(image):
    image
    return


@app.cell
def _(torchvision):
    pil_to_tensor = torchvision.transforms.PILToTensor()
    return (pil_to_tensor,)


@app.cell
def _(image, pil_to_tensor):
    input_uint8 = pil_to_tensor(image)
    input_uint8
    return (input_uint8,)


@app.cell
def _(input_uint8):
    input_uint8.shape
    return


@app.cell
def _(input_uint8):
    input_uint8.dtype
    return


@app.cell
def _(input_uint8):
    input = input_uint8 / (2**8 - 1)
    input
    return (input,)


@app.cell
def _(input):
    input.shape
    return


@app.cell
def _(input):
    input.dtype
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inference
    """)
    return


@app.cell
def _(input, model):
    model(input) # works
    return


@app.cell
def _(input, model, torch):
    # Better
    model.eval()
    with torch.no_grad():
        scores = model(input)
    scores
    return (scores,)


@app.cell
def _(scores):
    # "Winner" (max score)
    best_score_index = scores.argmax()
    best_score_index
    return (best_score_index,)


@app.cell
def _(best_score_index, dataset):
    dataset.classes[best_score_index]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uncertainty: Logits and Probabilities

    For a more subtle understanding of the results, the neural network outputs (or scores) can be interpreted as unnormalized log-probabilities or **logits**.

    Since a score $s$ can take any value in $[-\infty, +\infty]$, we could define a corresponding probability $p \in [0, 1]$ with

    $$
    \log p := s
    $$

    which amount to interpret scores are (normalized) log-probabilities. This is quite sound: the higher the score of a category, the higher its probability, a score equal to $-\infty$ corresponds to a 0 probability, a score of $+\infty$ corresponds to a probability of $1$, etc. However if we picked this convention, we would have necessarily

    $$
    \sum_{i=0}^9 e^{s_i} = \sum_{i=0}^9 p_i = 1
    $$

    while in reality, the output of the neural network satisfies this constraint.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So instead, we can use **logits**, or unnormalized log-probabilities, as score, that is

    $$
    \log p + c := s
    $$

    where the constant $c$ is determined by the score vector. Indeed, we have

    $$
    \sum_{i=0}^9 e^{s_i} = \sum_{i=0}^9 e^{\log p_i + c} = \left(\sum_{i=0}^9 p_i\right)e^{c} = e^c.
    $$

    With this convention, we have:

    $$
    \boxed{
    p_i = \frac{e^{s_i}}{e^c} = \frac{e^{s_i}}{\sum_{j=0}^9 e^{s_j}}.
    }
    $$

    The [softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html) operation computes the vector of all probabilities at once from the logits:

    $$
    \boxed{
    (p_0, \dots, p_9) = \mathrm{softmax}((s_0, \dots, s_9))
    }
    $$
    """)
    return


@app.cell
def _(scores):
    scores
    return


@app.cell
def _(scores, torch):
    softmax = torch.nn.Softmax(dim=1)
    p = softmax(scores).squeeze()
    p
    return p, softmax


@app.cell
def _(dataset, p, pd):
    df = pd.DataFrame(
        {
            "Category": dataset.classes,
            "Probability": p,
        }
    )
    df
    return (df,)


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(10, 2))
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    sns.barplot(x="Category", y="Probability", data=df)
    return


@app.cell(hide_code=True)
def _(best_score_index, cls, dataset, mo, p):
    mo.md(rf"""
    The real category **{dataset.classes[cls]}** and the neural networked inferred **{dataset.classes[best_score_index]}** with a confidence of **{p[best_score_index]*100:.0f} %**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Consolidation
    """)
    return


@app.cell
def _(model, pil_to_tensor, softmax, torch):
    def classify(image):
        input = pil_to_tensor(image) / (2**8 - 1)
        with torch.no_grad():
            logits = model(input)
        return softmax(logits).squeeze()

    return (classify,)


@app.cell
def _(dataset, pd, plt, sns):
    def visualize(probas):
        df = pd.DataFrame({ "Category": dataset.classes, "Probability": probas})
        plt.figure(figsize=(10, 2))
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        sns.barplot(x="Category", y="Probability", data=df)
        return plt.gcf()

    return (visualize,)


@app.cell
def _(classify, image, visualize):
    visualize(probas=classify(image))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Batched Prediction
    """)
    return


@app.cell
def _(dataset, pil_to_tensor, torch):
    images = []
    for _i, (_image, _) in enumerate(dataset):
        if _i >= 10:
            break
        images.append(pil_to_tensor(_image) / 255)
    images_tensor = torch.cat(images)
    images_tensor
    return (images_tensor,)


@app.cell
def _(images_tensor):
    images_tensor.shape
    return


@app.cell
def _(images_tensor, model, torch):
    with torch.no_grad():
        logits = model(images_tensor)
    logits
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Out-of-dataset samples
    """)
    return


@app.cell
def _(PIL):
    image_kiprun = PIL.Image.open("images/kiprun-28x28.png")
    image_kiprun
    return (image_kiprun,)


@app.cell
def _(classify, image_kiprun, visualize):
    probas_kiprun = classify(image_kiprun)
    visualize(probas_kiprun)
    return


@app.cell
def _(PIL):
    image_columbia = PIL.Image.open("images/columbia-28x28.png")
    image_columbia
    return (image_columbia,)


@app.cell
def _(classify, image_columbia, visualize):
    probas_columbia = classify(image_columbia)
    visualize(probas_columbia)
    return


if __name__ == "__main__":
    app.run()
