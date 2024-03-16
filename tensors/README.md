# Tensors

## Definitions

**TODO:** start with 1-dim and 2-dim tensors, higher-dim and 0-dim later (more complex).


**TODO:** formal definition with size and dim an data type(?) as a function of dim indices.

tensor $\simeq$ multi-dimensional array

  - 0-dim tensor $\simeq$ scalar
  - 1-dim tensor $\simeq$ vector
  - 2-dim tensor $\simeq$ matrix

0-dim tensor, shape $()$:

$$
1.0
$$

1-dim tensor, shape $(1)$:

$$
[1.0]
$$

1-dim tensor, shape $(3)$, in row form:


$$
\begin{bmatrix}
    1.0 & 2.0 & 3.0
\end{bmatrix}
$$

or if it's more convenient, in column form:

$$
\begin{bmatrix}
    1.0 \\ 
    2.0 \\
    3.0
\end{bmatrix}
$$

2-dim tensor, shape $(2, 3)$:

$$
\begin{bmatrix}
    \begin{bmatrix}
        1.0 & 2.0 & 3.0
    \end{bmatrix} 
        \begin{bmatrix}
        4.0 & 5.0 & 6.0
    \end{bmatrix} 
\end{bmatrix}
$$

or if it's more convenient in hybrid row-colum and column-row forms: 

$$
\begin{bmatrix}
    \begin{bmatrix}
        1.0 & 2.0
    \end{bmatrix} \\ 
    \begin{bmatrix}
        3.0 & 4.0
    \end{bmatrix} 
\end{bmatrix}
$$

$$
\begin{bmatrix}
    \begin{bmatrix}
        1.0 \\
        2.0 \\
        3.0 
    \end{bmatrix} 
    \begin{bmatrix}
        4.0 \\
        5.0 \\
        6.0
    \end{bmatrix} 
\end{bmatrix}
$$



> [!WARNING]  
> The usual matrix-form is ambiguous, don't use it (or at your own risk):
> 
> $$
> \begin{bmatrix}
>    1.0 & 2.0 & 3.0 \\ 
>    4.0 & 5.0 & 6.0 
> \end{bmatrix}
> $$
>

Of course, tensors shape don't stop here, you may have higher-dimensional tensors. Yes, there are concrete use cases for them!


## Pytorch

Preamble:

```python
import torch
```

```pycon
>>> t0 = torch.tensor(1.0)
>>> t0
tensor(1.)
```

```pycon
>>> t1 = torch.tensor([1.0, 2.0, 3.0])
>>> t1
tensor([1., 2., 3.])
```

```pycon
>>> t2 = torch.tensor([[1.0, 2.0, 3.0], 
...                    [4.0, 5.0, 6.0]])
>>> t2
tensor([[1., 2., 3.],
        [4., 5., 6.]])
```





## Tensor calculus

(everything is a contraction pretty-much)

## Vectorization 

(and speed)


## Examples (scalars, vectors, matrices, images, classifiers with proba, batches, etc.)


