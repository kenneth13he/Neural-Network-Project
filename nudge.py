from back_propagation import Value

def build(a_val, b_val, c_val, f_val):
    a = Value(a_val, label='a')
    b = Value(b_val, label='b')
    c = Value(c_val, label='c')
    f = Value(f_val, label='f')

    e = a*b; e.label='e'
    d = e + c; d.label='d'
    L = d*f; L.label='L'
    return L, a
    

h = 0.01

L1, a1 = build(2.0, -3.0, 10.0, -2.0)
L1.backward()

L2, _ = build(2.0 + h, -3.0, 10.0, -2.0)

predicted = a1.grad * h
actual = L2.data - L1.data

print("predicted:", predicted)
print("actual:   ", actual)

# ---------------------------------------------------------------
# BONUS (once the above works):
#   1. Nudge b instead of a. What should the prediction be? (check b.grad first)
#   2. Try h = 1. Do predicted and actual still match? Why or why not?
#   3. Try h = 0.0000001. What happens?
