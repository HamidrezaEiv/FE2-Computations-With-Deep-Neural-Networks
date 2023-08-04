import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as onp
import jax.numpy as jnp

import joblib

from utils import prep
from time import time

import jax
import optax

from flax import linen as nn
from flax.training import train_state

import tensorflow as tf
from tensorflow.data import Dataset as tfds

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices([], 'GPU')

nneur = 128
nhl = 8

test_name = 'NN_AD_JAX'
model_name = f'{nneur}_{nhl}_swish'

fdir = '../data/training_data.csv'

if not os.path.exists(f'./{test_name}'):
    os.mkdir(f'./{test_name}')
    
if not os.path.exists(f'./{test_name}/{model_name}'):
    os.mkdir(f'./{test_name}/{model_name}')

sdir = f'./{test_name}/{model_name}/'

pp = prep(fdir)
train, val = pp.scale(pp.normscaler)
joblib.dump(pp, sdir + 'pp')

X, y, y_x = train
X_v, y_v, y_x_v = val

nf = 3
nv = 3

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        def f(x):
            for i in range(8):
                x = nn.Dense(128)(x)
                x = nn.swish(x)
            return nn.Dense(nv)(x)
        jf = jax.jacrev(f)
        return f(x), jf(x).reshape((-1))
    
class LinenVmapMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        VmapMLP = nn.vmap(MLP, variable_axes={'params': None}, split_rngs={'params': False}, in_axes=0)
        return VmapMLP(name='mlp')(x)

model = LinenVmapMLP()
apply_fn = jax.jit(lambda params, inputs: model.apply({'params': params}, inputs))

@jax.jit
def train_step(state, X, y, y_x):
    @jax.jit
    def loss_fn(params):
        yp, y_xp = apply_fn(params, X)
        loss_1 = ((y - yp)**2)
        loss_2 = ((y_x - y_xp)**2)  * pp.coefs_J
        return loss_1.mean() + loss_2.mean()
    
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

@jax.jit
def eval_step(params, X, y, y_x):
    yp, y_xp = apply_fn(params, X)
    loss_1 = ((y - yp)**2)
    loss_2 = ((y_x - y_xp)**2)
    return loss_1.mean(), loss_2.mean()

n_batches = 100
batch_size = int(len(X) / n_batches)

data_train = tfds.from_tensor_slices((X, y, y_x))
data_test =  tfds.from_tensor_slices((X_v, y_v, y_x_v))
data_train = data_train.batch(batch_size, drop_remainder=False)

schedule = optax.exponential_decay(1e-3, 1000 * n_batches, 0.1, staircase=True)

key = jax.random.PRNGKey(32)
state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=model.init(key, jnp.empty((1, nf)))['params'],
      tx=optax.adam(schedule)
      )

hist = {'loss_y': [], 'val_loss_y': [], 'loss_yx': [], 'val_loss_yx': []}
start_of_training = time()
for epoch in range(4000):
    iterator = iter(data_train)
    strt_ep = time()
    for i, data_batch in enumerate(iterator):
        data_batch = [x.numpy() for x in data_batch]
        Xb, yb, y_xb = data_batch
        state = train_step(state, Xb, yb, y_xb)
        
    loss, loss_x = eval_step(state.params, X, y, y_x)
    val_loss, val_loss_x = eval_step(state.params, X_v, y_v, y_x_v)
    end_ep = time()
    time_ep = end_ep - strt_ep
    print('epoch {}: loss - {:.4e} - val_loss - {:.4e}, loss_x - {:.4e} - val_loss_x - {:.4e} - time - {:.3f}s'.format(
        epoch + 1, loss, val_loss, loss_x, val_loss_x, time_ep))
    hist['loss_y'].append(loss)
    hist['val_loss_y'].append(val_loss)
    hist['loss_yx'].append(loss_x)
    hist['val_loss_yx'].append(val_loss_x)
    

end_of_training = time()
cp_time = end_of_training - start_of_training

joblib.dump(state.params, sdir + 'params')
hist = onp.array(hist)
onp.savez_compressed(sdir + 'res', hist = hist, cp_time = cp_time)
