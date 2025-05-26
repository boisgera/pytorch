import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    import torchvision
    return torch, torchvision


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We download the subset of the FashionMNIST dataset that has *not* been used to train our model and put it in the `cache` directory.""")
    return


@app.cell
def _(torchvision):
    dataset = torchvision.datasets.FashionMNIST(root="cache", download=True, train=False)
    dataset
    return (dataset,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Any item in this dataset is a pair of [PIL] Image and class index.

    [PIL]: https://he-arc.github.io/livre-python/pillow/index.html
    """
    )
    return


@app.cell
def _(dataset):
    index = 0 # Select your item
    assert 0 <= index < len(dataset)
    return (index,)


@app.cell
def _(dataset, index):
    datum = dataset[index]
    datum
    return (datum,)


@app.cell
def _(datum):
    image, cls = datum
    return cls, image


@app.cell
def _(image):
    image
    return


@app.cell
def _(image):
    image.width, image.height
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`image.mode = 'L'` stands for 8-bit grayscale, see [PIL modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes).""")
    return


@app.cell
def _(image):
    image.mode
    return


@app.cell
def _(image):
    image.format_description
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The list of all object classes is available in `dataset.classes`. Use the class index as an index into this list to get the class name.""")
    return


@app.cell
def _(dataset):
    dataset.classes
    return


@app.cell
def _(cls, dataset):
    dataset.classes[cls] # Item category
    return


@app.cell
def _(dataset):
    dataset.classes # All known categories
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model Architecture""")
    return


@app.cell
def _(torch):
    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.linear_1 = torch.nn.Linear(28*28, 512)
            self.relu_1 = torch.nn.ReLU()
            self.linear_2 = torch.nn.Linear(512, 512)
            self.relu_2 = torch.nn.ReLU()
            self.linear_3 = torch.nn.Linear(512, 10)

        def forward(self, image_tensor):
            image_flat = self.flatten(image_tensor)
            x_0 = image_flat
            x_1 = self.linear_1(x_0)
            x_1 = self.relu_1(x_1)
            x_2 = self.linear_2(x_1)
            x_2 = self.relu_2(x_2)
            x_3 = self.linear_3(x_2)
            logits = x_3
            return logits
    return (NeuralNetwork,)


@app.cell
def _(NeuralNetwork):
    model = NeuralNetwork()
    model
    return (model,)


@app.cell
def _(model):
    model.linear_1.weight
    return


@app.cell
def _(model):
    model.state_dict()
    return


@app.cell
def _(model):
    for key, value in model.state_dict().items():
        print(key)
    return


@app.cell
def _(model):
    model.linear_1.weight
    return


@app.cell
def _(torch):
    state_dict = torch.load("models/base-model.pth")
    state_dict
    return (state_dict,)


@app.cell
def _(model, state_dict):
    model.load_state_dict(state_dict)
    model
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Transformation""")
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
    mo.md(r"""## Inference""")
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
    mo.md(
        r"""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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

    """
    )
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
    return (p,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


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
def _(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 2))
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    sns.barplot(x="Category", y="Probability", data=df)
    return


@app.cell(hide_code=True)
def _(best_score_index, cls, dataset, mo, p):
    mo.md(rf"""The real category **{dataset.classes[cls]}** and the neural networked inferred **{dataset.classes[best_score_index]}** with a confidence of **{p[best_score_index]*100:.0f} %**.""")
    return


if __name__ == "__main__":
    app.run()
