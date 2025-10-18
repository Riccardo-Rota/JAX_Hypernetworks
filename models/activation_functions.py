import flax.nnx as nnx

def get_tanh():
    return nnx.tanh

def get_relu():
    return nnx.relu

def get_sigmoid():
    return nnx.sigmoid

def get_gelu():
    return nnx.gelu