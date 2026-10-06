from graphviz import Digraph  # Digraph = a "directed graph" (arrows point one way)

def trace(root):
    # builds a set of all nodes and edges of a graph
    nodes, edges = set(), set()  # sets so nothing gets added twice

    def build(v):
        if v not in nodes:  # skip values we've already visited
            nodes.add(v)  # remember this value as a node
            for child in v._prev:  # look at every value that was used to make v
                edges.add((child, v))  # add an arrow: child -> v
                build(child)  # repeat for the child (walks all the way back)

    build(root)  # start from the final value and walk backwards
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR is left to right

    nodes, edges = trace(root)  # get every value and every connection
    for n in nodes:
        uid = str(id(n))  # id() gives each object a unique number, used as its name in the graph

        # for each value in the graph, create a rectangle node
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

        if n._op:  # if this value was made by an operation (+ or *)
            dot.node(name=uid + n._op, label=n._op)  # create a small oval node for the operation
            dot.edge(uid + n._op, uid)  # arrow from the operation to the value it produced

    for n1, n2 in edges:
        # arrow from the input value (n1) into the operation that made n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
