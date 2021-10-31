import cupy as cp

#===========================
# softmax
#===========================
def softmax(x):
    x = x - cp.max(x, axis=-1, keepdims = True)
    out = cp.exp(x) / cp.sum(cp.exp(x), axis=-1, keepdims = True)

    return out

#============================
# relu
#============================
def relu(x):
    mask = (x <= 0)
    out = x.copy()
    out[mask] = 0

    return out, mask

#============================
#  sigmoid
#============================
def sigmoid(x):
    return 1 / (1 + cp.exp(-x))
