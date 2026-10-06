# Neural Network Project

A tiny neural network library built from scratch in Python, with diagrams of how values and gradients flow through each calculation.

The goal is to turn this into an **interactive learning tool**, where you can change values and watch backpropagation update in real time. That part is still being built; right now the project has the engine, the neural network, and static graph drawing.

## What's inside

| File | What it does |
|---|---|
| `back_propagation.py` | `Value`, a number that remembers how it was made and can compute its own gradient with `.backward()` |
| `neural_network.py` | `Neuron`, `Layer`, and `MLP`, a small neural network built out of `Value`s |
| `draw_dot.py` | Draws the calculation graph, showing each value's data and gradient |

## Setup

You need Python 3 and the Graphviz program (used to draw the graphs).

```bash
# install Graphviz (Mac)
brew install graphviz

# install the Python package
python3 -m pip install -r requirements.txt
```

On Windows or Linux, get Graphviz from [graphviz.org/download](https://graphviz.org/download/).

## Usage

**Draw a calculation and its gradients:**

```bash
python3 back_propagation.py
```

This builds `L = (a*b + c) * f`, runs backpropagation, and opens `graph.svg` showing every value and its gradient.

**Use it in your own code:**

```python
from back_propagation import Value
from draw_dot import draw_dot

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a * b; c.label = 'c'

c.backward()
print(a.grad)  # -3.0: increasing a by 1 changes c by about -3

draw_dot(c).render('graph', view=True)
```

**Train a small network:**

```python
from neural_network import MLP

model = MLP(3, [4, 4, 1])  # 3 inputs -> 4 neurons -> 4 neurons -> 1 output
xs = [[2, 3, -1], [3, -1, 0.5], [0.5, 1, 1], [1, 1, -1]]
ys = [1, -1, -1, 1]

for step in range(50):
    loss = sum((model(x) - y)**2 for x, y in zip(xs, ys))
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad
    print(step, loss.data)
```

## Credit

Based on [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy (MIT license), and his video [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0).
