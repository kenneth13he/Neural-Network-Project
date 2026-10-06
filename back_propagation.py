import math

# based on micrograd by Andrej Karpathy (MIT license): https://github.com/karpathy/micrograd

class Value:
    # initializing the class
    # itself, data, what values created it and the operation that underwent
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # how much the final output changes if this value is nudged
        self._backward = lambda: None  # function that passes the gradient to the children
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # remaking the print statement
    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

    # creating the addition function
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # lets you do a + 1
        out = Value(self.data + other.data, (self, other), '+')

        # addition passes the gradient straight through to both inputs
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    # creating multiplication function
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # lets you do a * 2
        out = Value(self.data * other.data, (self, other), '*')

        # each input's gradient is the other input's value times the output gradient
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    # creating the power function (only numbers as the power, like a**2)
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        # power rule: d/dx x^n = n * x^(n-1)
        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward

        return out

    # creating the exponential function e^x
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        # the derivative of e^x is e^x
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    # creating tanh, squashes any number into the range -1 to 1
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        # the derivative of tanh(x) is 1 - tanh(x)^2
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    # creating relu, turns negatives into 0
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        # gradient only flows through if the value was positive
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    # runs backpropagation from this value through the whole graph
    def backward(self):
        # every value comes after the values that made it
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one value at a time from the output backwards and apply the chain rule
        self.grad = 1.0  # the output's gradient with respect to itself is 1
        for v in reversed(topo):
            v._backward()

    # the rest are built from the operations above
    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self (like 1 + a)
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self (like 2 * a)
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1


# only runs when you run this file directly
if __name__ == '__main__':
    from draw_dot import draw_dot  # imported here so Value works without graphviz (like on the website)

    # test inputs
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a*b; e.label='e'
    d = e + c; d.label='d'
    f = Value(-2.0, label='f')
    L = d*f; L.label='L'

    L.backward()  # fills in .grad for every value
    draw_dot(L).render('graph', view=True)   # saves graph.svg and opens it
