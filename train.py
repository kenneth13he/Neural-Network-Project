import random
from neural_network import MLP
import math

def make_dataset(n):
    dots = []
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        label = 1 if x * y > 0 else -1
        dots.append((x, y, label))
    return dots


def make_model(x: int, y: list):
    return MLP(x, y)


def train_step(model, dots, lr):
    loss = 0
    for x, y, label in dots:
        pred = model([x, y])
        loss += (pred - label)**2
    loss = loss/len(dots)

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data = p.data - lr * p.grad
    return loss.data


def predict(model, x, y):
    pred = model([x, y])
    return pred.data


def predict_fast(model, x, y):
    inputs = [x, y]
    for layer in model.layers:
        outputs = []
        for neuron in layer.neurons:
            total = neuron.b.data
            for w, inp in zip(neuron.w, inputs):
                total += w.data * inp
            if neuron.nonlin:
                total = math.tanh(total)
            outputs.append(total)
        inputs = outputs
    return inputs[0]


def accuracy(model, dots):
    correct = 0
    for x, y, label in dots:
        guess = predict(model, x, y)
        if guess * label > 0:
            correct += 1
    return correct/len(dots)


if __name__ == '__main__':
    dots = make_dataset(40)
    model = make_model(2, [4, 4, 1])
    lr = 0.2

    for step in range(400):
        loss = train_step(model, dots, lr)

        if step % 20 == 0:
            print('step', step, 'loss', round(loss, 4), 'accuracy', accuracy(model, dots))
