import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib
import jax.numpy as jnp
import jax
from flax import linen as nn

print('Loading models ...')
models_dir = os.getcwd() + '/../trained_models/NN_AD_JAX/128_8_swish/'
print(models_dir)
pp = joblib.load(models_dir + 'pp')
params = joblib.load(models_dir + 'params')

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        def f(x):
            for i in range(8):
                x = nn.Dense(128)(x)
                x = nn.swish(x)
            return nn.Dense(3)(x)
        jf = jax.jacobian(f)
        return f(x), jf(x).reshape((-1))

class LinenVmapMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        VmapMLP = nn.vmap(MLP, variable_axes={'params': None}, split_rngs={'params': False}, in_axes=0)
        return VmapMLP(name='mlp')(x)

model = LinenVmapMLP()
print('Done!')

@jax.jit
def prediction(*args):
    E = jnp.array(args)
    E = E.reshape((-1, 3))
    E = pp.scale_x(E)
    T, C = jax.jit(model.apply)({'params': params}, E)

    E, T, C = pp.scale_r((E, T, C))

    tc = ()
    for t in T[0]:
        tc = tc + (t,)
        
    for c in C[0]:
        tc = tc + (c,)
    return tc
