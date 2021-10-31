import cupy as cp

#====================================
# cross_entropy
#====================================
def cross_entropy_error(y, t):
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size

#====================================
# binary_cross_entropy
#====================================
def binary_cross_entropy(y, t):
    delta = 1e-6
    return -cp.mean(t * cp.log(y + delta) + (1 - t) * cp.log(1 - y + delta))