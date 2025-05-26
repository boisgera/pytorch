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
    # Training

    Source: [Pytorch tutorials](https://docs.pytorch.org/tutorials/)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    /// note | Learning Objectives

    - [ ] Understand the structure of the FashionMNIST dataset
    - [ ] Represent its contents appropriately
    - [ ] Understand the purpose of the dataset loading options
    ///

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// note | Learning Objectives
    - [ ] Load the pre-trained neural network model
    - [ ] Understand its output
    - [ ] Use it for classification
    - [ ] Use it in batched mode
    - [ ] Evaluate the confidence in its answers
    - [ ] Understand the components it uses
    - [ ] Get the model parameters
    - [ ] Replicate manually its computation
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// Note | Learning Objectives
    - [ ] Understand the loss function concept
    - [ ] Understand what cross-entropy computes
    - [ ] Understand what data loader does (at a high-level)
    - [ ] Understand what the training loop does (at a high-level)
    - [ ] Estimate model accuracy
    - [ ] Save a model
    ///
    """
    )
    return


@app.cell
def _():
    import torch
    import torchvision
    return torch, torchvision


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The FashionMNIST Dataset""")
    return


@app.cell
def _(torchvision):
    # Download training data from open datasets.
    training_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # Download test data from open datasets.
    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return test_data, training_data


@app.cell
def _(torchvision):
    # by default: training data set, no input/output transform, no download
    data = torchvision.datasets.FashionMNIST(root="data") 
    data
    return (data,)


@app.cell
def _(data):
    # data is list-like ; each item in an input-output pair
    datum = data[0]
    datum
    return (datum,)


@app.cell
def _(datum):
    image, index = datum
    return image, index


@app.cell
def _(image):
    image
    return


@app.cell
def _(data):
    # The output is a number that denotes the class of the pictured object. 
    # The list of classes is:
    data.classes
    return


@app.cell
def _(data, index):
    # Get the category name from the index:
    data.classes[index] # that checks out!
    return


@app.cell
def _(data):
    import pandas as pd
    df = [{"image": image, "class": data.classes[index]} for image, index in data]
    df = pd.DataFrame(df)
    df
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(image, torchvision):
    # Pytorch only want to deal with numeric array called "tensors", not images.
    # So, we need to transform the input
    image_to_tensor = torchvision.transforms.ToTensor()
    t = image_to_tensor(image)
    t
    return (t,)


@app.cell
def _(t):
    t.shape, t.dtype
    return


@app.cell
def _(t):
    # No information has been lost in the conversion process!
    import matplotlib.pyplot as plt
    plt.imshow(t.squeeze(), cmap="grey")
    plt.colorbar()
    plt.gcf()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Neural Network Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The model architecture is going to assign to each image input the vector of probability $p_i$ that the item belongs to the $i$th class.
    The class prediction is simply the class with the highest probability, but the fact that all $p_i$ are known allows us to evaluate the trust that we should have in the prediction.

    The nitty-gritty details: the model does actually not output the probabilities $p_i \in [0, 1]$ directly but the corresponding unnormalized log probabilities

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
    """
    )
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
def _(model, torch):
    # Load the (trained) model state for this architecture
    model.load_state_dict(torch.load("models/base-model.pth"))
    return


@app.cell
def _(plt, training_data):
    image_tensor, cls = training_data[0]
    plt.imshow(image_tensor.squeeze(), cmap="grey")
    plt.grid(False)
    None
    return (image_tensor,)


@app.cell
def _(image_tensor, model, torch):
    with torch.no_grad():
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
    probas_dict = {training_data.classes[i]: _p.item() for i, _p in enumerate(probas)}
    probas_dict
    return (probas_dict,)


@app.cell
def _(plt, probas_dict):
    import seaborn as sns; sns.set_theme()
    sns.barplot(probas_dict)
    plt.gcf().set_figwidth(12)
    plt.gca().set_ylabel("Probability")
    None
    return (sns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Batched Prediction""")
    return


@app.cell
def _(torch, training_data):
    images = []
    for i, (image_1, _) in enumerate(training_data):
        if i >= 10:
            break
        images.append(image_1)
    images_tensor = torch.cat(images)
    images_tensor
    return image_1, images_tensor


@app.cell
def _(images_tensor, model, torch):
    with torch.no_grad():
        output = model(images_tensor)
    output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Parameters""")
    return


@app.cell
def _(model):
    for _p in model.parameters():
        print(f'type: {type(_p.data).__name__}, shape: {tuple(_p.shape)!s:<10}, data type: {_p.dtype}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```{tip} Model Size
    How many scalar parameters describe the model? What is the corresponding model size in MB?
    ```
    """
    )
    return


@app.cell
def _(model, torch):
    num_params = 0
    for _p in model.parameters():
        num_params = num_params + torch.prod(torch.tensor(_p.shape)).item()
    return (num_params,)


@app.cell(hide_code=True)
def _(mo, num_params):
    mo.md(rf"""There are {num_params} ($\approx$ {num_params // 1_000}K) parameters in the model. The size of each parameter is 4B, hence the total size is {round(num_params * 4 / 1_000_000, 1)}MB.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Under the Hood""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Flatten""")
    return


@app.cell
def _(model):
    model.flatten
    return


@app.cell
def _(image_1):
    t_1 = image_1
    t_1
    return (t_1,)


@app.cell
def _(model, t_1):
    model.flatten(t_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Linear (Affine) Operator""")
    return


@app.cell
def _(model):
    model.linear_1
    return


@app.cell
def _(torch):
    linop = torch.nn.Linear(1, 2)
    linop.weight, linop.bias
    return (linop,)


@app.cell
def _(linop):
    list(linop.parameters())
    return


@app.cell
def _(torch):
    input = torch.linspace(0.0, 5.0, 6)
    input
    return (input,)


@app.cell
def _(input, linop, torch):
    with torch.no_grad():
        input_t = input.reshape(1, -1, 1)
        output_t = linop(input_t)
    output_1 = output_t.squeeze()
    return (output_1,)


@app.cell
def _(input, output_1, plt):
    plt.plot(input, output_1)
    None
    return


@app.cell
def _(model):
    model.linear_1.in_features == 28 * 28
    return


@app.cell
def _(model):
    linear_1_params = {name: param.data for name, param in model.linear_1.named_parameters()}
    linear_1_params
    return (linear_1_params,)


@app.cell
def _(linear_1_params):
    b1 = linear_1_params["bias"]
    b1.shape
    return (b1,)


@app.cell
def _(linear_1_params):
    A1 = linear_1_params["weight"]
    A1.shape
    return (A1,)


@app.cell
def _(A1, b1, image_1, torch):
    t_2 = image_1
    print(t_2.shape)
    tf = torch.flatten(t_2)
    print(tf.shape)
    x1 = A1 @ tf + b1
    x1
    return t_2, x1


@app.cell
def _(plt, x1):
    plt.imshow(x1.reshape((2**5, 2**4)))
    plt.grid(False)
    plt.colorbar()
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Rectified Linear Unit""")
    return


@app.cell
def _(model, x1):
    x1_1 = model.relu_1(x1)
    x1_1
    return (x1_1,)


@app.cell
def _(plt, torch):
    relu = torch.nn.ReLU()
    input_1 = torch.linspace(-3.0, 3.0, 7)
    relu(input_1)
    plt.plot(input_1, relu(input_1))
    plt.axis('equal')
    None
    return


@app.cell
def _(plt, x1_1):
    plt.imshow(x1_1.reshape((2 ** 5, 2 ** 4)))
    plt.grid(False)
    plt.colorbar()
    None
    return


@app.cell
def _(model):
    list(model.relu_1.named_parameters())
    return


@app.cell
def _(model, x1_1):
    linear_2_params = {name: param.data for name, param in model.linear_2.named_parameters()}
    A2 = linear_2_params['weight']
    b2 = linear_2_params['bias']
    x2 = A2 @ x1_1 + b2
    x2
    return (x2,)


@app.cell
def _(plt, x2):
    plt.imshow(x2.reshape((2**5, 2**4)))
    plt.grid(False)
    plt.colorbar()
    None
    return


@app.cell
def _(model, x2):
    x2_1 = model.relu_2(x2)
    x2_1
    return (x2_1,)


@app.cell
def _(plt, x2_1):
    plt.imshow(x2_1.reshape((2 ** 5, 2 ** 4)))
    plt.grid(False)
    plt.colorbar()
    None
    return


@app.cell
def _(model, x2_1):
    linear_3_params = {name: param.data for name, param in model.linear_3.named_parameters()}
    A3 = linear_3_params['weight']
    b3 = linear_3_params['bias']
    x3 = A3 @ x2_1 + b3
    x3
    return (x3,)


@app.cell
def _(torch, x3):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=0)
    probas_1 = softmax(x3)
    probas_1
    return (probas_1,)


@app.cell
def _(probas_1, training_data):
    data_1 = {training_data.classes[i]: _p.item() for i, _p in enumerate(probas_1)}
    data_1
    return (data_1,)


@app.cell
def _(data_1, plt, sns):
    sns.barplot(data_1)
    plt.gcf().set_figwidth(12)
    plt.gca().set_ylabel('Probability')
    None
    return


@app.cell
def _(model, t_2):
    out = model(t_2).squeeze()
    return (out,)


@app.cell
def _(out, torch):
    ps = torch.nn.functional.softmax(out, dim=-1)
    return (ps,)


@app.cell
def _(plt, ps, sns, training_data):
    data_2 = {training_data.classes[i]: _p.item() for i, _p in enumerate(ps)}
    sns.barplot(data_2)
    plt.gcf().set_figwidth(12)
    plt.gca().set_ylabel('Probability')
    None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Training""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Loss Function""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The loss function is a measure of the model prediction error: the mismatch between the output predicted by the model and the "real" output. Here, in the context of category identification, we use the cross-entropy loss function.""")
    return


@app.cell
def _(torch):
    _loss_function = torch.nn.CrossEntropyLoss()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    It measures the difference between two probability distributions: here a computed probability distribution $p=(p_0, \dots, p_{n-1})$ and the "deterministic" distribution $e_i$ that states that the $i$ event is certain: every entry of $e_i$ is null but the $i$th which is equal to 1:

    $$
    e_i=(0, \dots, 0, 1, 0, \dots, 0)
    $$

    The difference is measured with:

    $$
    \mathrm{loss}(p, e_i) = - \sum_{j} e_{ij} \log p_j = - \log p_i. 
    $$

    The loss is zero when $p_i = 1$ (perfect match) and $-\infty$ when $p_i = 0$. It does not depend on the distribution of the $p_j$ for $j \neq i$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```{warning}
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
    ```
    """
    )
    return


@app.cell
def _(torch):
    cross_entropy = torch.nn.CrossEntropyLoss()
    return (cross_entropy,)


@app.cell
def _(cross_entropy, torch):
    cross_entropy(
        input=torch.tensor([1.0, 0.0]).log(), 
        target=torch.tensor(0)
    )
    return


@app.cell
def _(cross_entropy, torch):
    cross_entropy(
        input=torch.tensor([0.0, 1.0]).log(), 
        target=torch.tensor(0)
    )
    return


@app.cell
def _(cross_entropy, torch):
    cross_entropy(
        input=torch.tensor([0.5, 0.5]).log(), 
        target=torch.tensor(0)
    )
    return


@app.cell
def _(torch):
    - torch.tensor(0.5).log()
    return


@app.cell
def _(cross_entropy, torch):
    cross_entropy(
        input=torch.tensor([2/3, 1/3]).log(), 
        target=torch.tensor(0)
    )
    return


@app.cell
def _(cross_entropy, torch):
    cross_entropy(
        input=torch.tensor([200.0, 100.0]).log(), 
        target=torch.tensor(0)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Data Loader""")
    return


@app.cell
def _(test_data, torch, training_data):
    batch_size = 64

    # Create data loaders.
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"X = [N, C, H, W]: {X.shape}")
        print(f"y: {y.shape} {y.dtype}")
        break
    return test_dataloader, train_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Training""")
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
                correct = correct + (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss = test_loss / num_batches
        correct = correct / size
        print(f'Test Error: \n Accuracy: {100 * correct:>0.2f}%, Avg loss: {test_loss:>8f} \n')
        return correct
    return (test,)


@app.cell
def _(NeuralNetwork, test, test_dataloader, torch, train_dataloader):
    model_1 = NeuralNetwork()
    model_1.train()
    optimizer = torch.optim.SGD(model_1.parameters(), lr=0.001)
    _loss_function = torch.nn.CrossEntropyLoss()
    score = 0.0
    new_score = test(test_dataloader, model_1, _loss_function)
    epoch = 0
    keep_learning = True
    while keep_learning:
        epoch = epoch + 1
        score = new_score
        print(f'Epoch {epoch + 1}\n-------------------------------')
        train(train_dataloader, model_1, _loss_function, optimizer)
        new_score = test(test_dataloader, model_1, _loss_function)
        keep_learning = new_score > score
    print('Done!')
    return (model_1,)


@app.cell
def _(mo):
    mo.md(r"""## Saving the Model""")
    return


@app.cell
def _(model_1, torch):
    torch.save(model_1.state_dict(), 'models/model.pth')
    print('Saved PyTorch Model State to model.pth')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
