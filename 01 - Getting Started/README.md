
```shell
boisgera@wreck:~/tmp/pytorch$ pixi init
✔ Created /home/boisgera/tmp/pytorch/pixi.toml
boisgera@wreck:~/tmp/pytorch$ pixi add python pytorch torchvision
✔ Added python >=3.13.3,<3.14                                           ✔ Added pytorch >=2.6.0,<3                                              ✔ Added torchvision >=0.21.0,<0.22
```

```pycon
>>> import torch.cuda
>>> torch.cuda.is_available()
False
```

```pycon
>>> import torch
>>> t = torch.tensor([1.0, 2.0, 3.0])
>>> t
tensor([1., 2., 3.])
>>> t.device
device(type='cpu')
```

-----

```shell
boisgera@wreck:~$ nvidia-smi
Mon May 19 16:17:27 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro P2000                   Off |   00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8             N/A / ERR!  |       9MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      3408      G   /usr/lib/xorg/Xorg                              4MiB |
+-----------------------------------------------------------------------------------------+
```

```toml
[system-requirements]
cuda = "12.4"
```


```shell
$ pixi remove pytorch
$ pixi add pytorch-gpu
```

```pycon
>>> import torch
>>> torch.cuda.is_available()
True
>>> tensor([1, 2, 3]).device
>>> torch.tensor([1, 2, 3]).device
device(type='cpu')
>>> torch.cuda.current_device()
0
>>> torch.cuda.get_device_name(0)
'Quadro P2000'
>>> torch.cuda.device_count()
1
```

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ...
model = NeuralNetwork()
model = model.to(device)
# ...
for batch, (X, y) in enumerate(dataloader):
    X = X.to(device)
    y = y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    # ...
```