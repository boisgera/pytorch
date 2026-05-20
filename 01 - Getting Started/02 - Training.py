import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training

    [Sébastien Boisgérault], Mines Paris - PSL University

    Source: [Pytorch tutorials](https://docs.pytorch.org/tutorials/)

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// Note | Learning Objectives

    **Everything is a Tensor**

    - [ ] Understand the principle of conversion of images as tensors,
    - [ ] Know why the single-precision floating-point numbers are used,

    **Model Architecture**

    - [ ] Learn how to assemble a model as a sequence of elementary models,
    - [ ] Understand the basic components that compose a neural network,

    **Loss Function**

    - [ ] Understand the concept of loss function,
    - [ ] Understand what cross-entropy computes and why it's appropriate here,

    **Training**

    - [ ] Understand what data loader does (at a high-level),
    - [ ] Understand what the training loop does (at a high-level),
    - [ ] Learn how to estimate model accuracy,
    - [ ] Learn how to read and save the model parameters,
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

    return pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The FashionMNIST Dataset
    """)
    return


@app.cell
def _(torchvision):
    # Download training data
    training_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # Download test data
    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return test_data, training_data


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
        torch.nn.Linear(28 * 28, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )

    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model architecture is going to assign to each image input the vector of probability $p_i$ that the item belongs to the $i$th class.
    The class prediction is simply the class with the highest probability, but the fact that all $p_i$ are known allows us to evaluate the trust that we should have in the prediction.

    The nitty-gritty details: the model does actually not output the probabilities $p_i \in [0, 1]$ directly but the corresponding unnormalized log probabilities (or logits)

    $$
    \ell_i := \log p_i + c
    $$

    because any value in $\mathbb{R}^{10}$ can be interpreted as vector of unnormalized log probabilies, this parameter is "free", devoid of constraints, in $\mathbb{R}^{10}$, so every possible output of or neural network is a valid value,
    while the vector of probabilities is constrained.

    If needed, compute $p_i$ with:

    $$
    p_i = \frac{\exp \ell_i}{\sum_{j=0}^{9} \exp \ell_j}.
    $$

    The pytorch [`softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) function implements this operation.
    """)
    return


@app.cell
def _(plt, training_data):
    image_tensor, cls = training_data[0]
    plt.imshow(image_tensor.squeeze(), cmap="grey")
    plt.grid(False)
    plt.colorbar()
    plt.gcf()
    return (image_tensor,)


@app.cell
def _(image_tensor, model):
    logits = model(image_tensor)
    logits = logits.squeeze()
    logits
    return (logits,)


@app.cell
def _(logits, torch):
    probas = torch.nn.functional.softmax(logits, dim=-1)
    probas
    return (probas,)


@app.cell
def _(probas, training_data):
    probas_dict = {
        training_data.classes[i]: _p.item() for i, _p in enumerate(probas)
    }
    probas_dict
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model State (Parameters)
    """)
    return


@app.cell
def _(model):
    model.state_dict()
    return


@app.cell
def _(model):
    list(model.parameters())
    return


@app.cell
def _(model):
    for _p in model.parameters():
        print(
            f"type: {type(_p.data).__name__}, shape: {tuple(_p.shape)!s:<10}, data type: {_p.dtype}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Model Size
    How many scalar parameters describe the model? What is the corresponding model size in MB?
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo, model, torch):
    num_params = 0
    for _p in model.parameters():
        num_params = num_params + torch.prod(torch.tensor(_p.shape)).item()

    mo.md(
        rf"""
    /// hint | Solution
    There are {num_params} ($\approx$ {num_params // 1_000}K) parameters in the model. 

    The size of each parameter is 4B, hence the total size is {round(num_params * 4 / 1_000_000, 1)}MB.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Components
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Flatten
    """)
    return


@app.cell
def _(model):
    flatten = model[0]
    flatten
    return (flatten,)


@app.cell
def _(image_tensor):
    image_tensor
    return


@app.cell
def _(flatten, image_tensor):
    image_vector = flatten(image_tensor)
    image_vector
    return (image_vector,)


@app.cell
def _(image_tensor, image_vector):
    print("before flatten:", image_tensor.shape)
    print("after flatten:", image_vector.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Linear (Affine) Operator
    """)
    return


@app.cell
def _(model):
    linear_1 = model[1]
    linear_1
    return (linear_1,)


@app.cell
def _(linear_1):
    list(linear_1.parameters())
    return


@app.cell
def _(linear_1):
    dict(linear_1.named_parameters())
    return


@app.cell
def _(linear_1):
    linear_1.state_dict()
    return


@app.cell
def _(linear_1):
    linear_1.weight.shape
    return


@app.cell
def _(linear_1):
    linear_1.bias.shape
    return


@app.cell
def _(image_vector, linear_1):
    image_vector_lin_1 = linear_1(image_vector)
    image_vector_lin_1
    return (image_vector_lin_1,)


@app.cell
def _(image_vector, linear_1):
    linear_1.weight @ image_vector.squeeze() + linear_1.bias
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rectified Linear Unit
    """)
    return


@app.cell
def _(model):
    relu = model[2]
    return (relu,)


@app.cell
def _(relu):
    relu
    return


@app.cell
def _(image_vector_lin_1):
    image_vector_lin_1
    return


@app.cell
def _(image_vector_lin_1, relu):
    relu(image_vector_lin_1)
    return


@app.cell
def _(plt, torch):
    def _():
        relu = torch.nn.ReLU()
        input_1 = torch.linspace(-3.0, 3.0, 7)
        relu(input_1)
        plt.plot(input_1, relu(input_1))
        plt.axis("equal")
        return plt.gcf()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loss Function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The loss function is a measure of the model prediction error: the mismatch between the output predicted by the model and the "real" output. Here, in the context of category identification, we use the cross-entropy loss function.
    """)
    return


@app.cell
def _(torch):
    loss_function = torch.nn.CrossEntropyLoss()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It measures the difference between two probability distributions: here a computed probability distribution $p=(p_0, \dots, p_{n-1})$ and the "deterministic" distribution $e_i$ that states that the $i$ event is certain: every entry of $e_i$ is null but the $i$th which is equal to 1:

    $$
    e_i=(0, \dots, 0, 1, 0, \dots, 0)
    $$

    The difference is measured with:

    $$
    \mathrm{loss}(p, e_i) = - \sum_{j} e_{ij} \log p_j = - \log p_i.
    $$

    The loss is zero when $p_i = 1$ (perfect match) and $-\infty$ when $p_i = 0$. It does not depend on the distribution of the $p_j$ for $j \neq i$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// warning

    The pytorch cross entropy function works directly with unnormalized log probabilities

    $$
    \ell_i := \log p_i + c
    $$

    (the log probabilities up to a shared constant $c$) instead of the probabilites $p$.
    The deterministic distribution is also specified by the index $i$ instead of the vector $q=e_i$.
    Hence, it actually computes

    $$
    \mathrm{loss}(\ell, i) := -\ell_i  + \log \left( \sum_je^{\ell_j} \right).
    $$

    ///
    """)
    return


@app.cell
def _(torch):
    cross_entropy = torch.nn.CrossEntropyLoss()
    return (cross_entropy,)


@app.cell
def _(cross_entropy, torch):
    cross_entropy(input=torch.tensor([1.0, 0.0]).log(), target=torch.tensor(0))
    return


@app.cell
def _(cross_entropy, torch):
    cross_entropy(input=torch.tensor([0.0, 1.0]).log(), target=torch.tensor(0))
    return


@app.cell
def _(cross_entropy, torch):
    cross_entropy(input=torch.tensor([0.5, 0.5]).log(), target=torch.tensor(0))
    return


@app.cell
def _(torch):
    -torch.tensor(0.5).log()
    return


@app.cell
def _(cross_entropy, torch):
    cross_entropy(input=torch.tensor([2 / 3, 1 / 3]).log(), target=torch.tensor(0))
    return


@app.cell
def _(cross_entropy, torch):
    cross_entropy(input=torch.tensor([200.0, 100.0]).log(), target=torch.tensor(0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Loader
    """)
    return


@app.cell
def _(test_data, torch, training_data):
    batch_size = 64

    # Create data loaders.
    train_dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"X = [N, C, H, W]: {X.shape}")
        print(f"y: {y.shape} {y.dtype}")
        break
    return test_dataloader, train_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training
    """)
    return


@app.function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


@app.cell
def _(torch):
    def test(dataloader, model, loss_fn):
        """
        Returns:
          - score, the probability of a correct inference on the test dataset
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = (0, 0)
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss = test_loss + loss_fn(pred, y).item()
                correct = (
                    correct + (pred.argmax(1) == y).type(torch.float).sum().item()
                )
        test_loss = test_loss / num_batches
        correct = correct / size
        print(
            f"Test Error: \n Accuracy: {100 * correct:>0.2f}%, Avg loss: {test_loss:>8f} \n"
        )
        return correct

    return (test,)


@app.cell
def _(model, test, test_dataloader, torch, train_dataloader):
    def learn(max_epoch=10, lr=1e-3):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_function = torch.nn.CrossEntropyLoss()

        score = 0.0
        new_score = test(test_dataloader, model, loss_function)
        epoch = 0

        while True:
            if epoch >= max_epoch or new_score <= score:
                break
            epoch = epoch + 1
            score = new_score
            print(f"Epoch {epoch}\n-------------------------------")
            train(train_dataloader, model, loss_function, optimizer)
            new_score = test(test_dataloader, model, loss_function)
        print("Done!")


    learn(lr=1e-2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving the Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | TODO
        type: warning

    - document the shortcomings of ".pth" format
    - safetensor export
    - ONNX export
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model’s `parameters` method returns an iterator over the model parameters as PyTorch tensors.
    """)
    return


@app.cell
def _(model):
    model.parameters()  # "lazy" list
    return


@app.cell
def _(model):
    list(model.parameters())  # true list
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model also has a `state_dict` method that name the same parameters:
    """)
    return


@app.cell
def _(model, pd):
    pd.DataFrame(
        [
            {
                "name": name,
                "shape": tuple(tensor.shape),
                "data type": str(tensor.dtype),
                "value": str(tensor),
            }
            for name, tensor in model.state_dict().items()
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simplest way to save the models parameters is with `torch.save`:
    """)
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), "models/model.pth")
    print("Saved PyTorch Model State to 'models/model.pth'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, this method relies on the `pickle` module from the Python standard library which can vary from one Python version to another. It is also tied to Pytorch.

    For more robustness and portability, you can use the [safetensors] library instead.

    [safetensors]: https://huggingface.co/docs/safetensors/index
    """)
    return


@app.cell
def _(model):
    import safetensors.torch

    safetensors.torch.save_file(model.state_dict(), "models/model.safetensors")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In both cases, only the model parameters are saved. To use the model to perform inference, you will need to provide the model architecture definition.

    With the [ONNX] (Open Neural Network Exchange format, both the model parameters and architecture are saved and you case use the model with any ONNX runtime, without Pytorch.

    [ONNX]: https://onnx.ai/
    """)
    return


@app.cell
def _(model, torch, train_dataloader):
    (example_input, example_output) = next(iter(train_dataloader))
    assert example_input.shape == torch.Size([64, 1, 28, 28])

    torch.onnx.export(
        model,
        example_input,
        "models/model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={
            "input": {0: "batch_size"},
        },
    )
    return


if __name__ == "__main__":
    app.run()
