import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras import models
import joblib

print('Loading models ...')
models_dir = os.getcwd() + '/../trained_models/NN_AD_TF/128_8_swish/'
print(models_dir)
pp = joblib.load(models_dir + 'pp')

model = models.load_model(models_dir + 'model', compile=False)
print('Done!')

def prediction(*args):
    E = np.array(args)
    E = E.reshape((1, -1))
    E = pp.scale_x(E)
    T, C = model(E, training=False)
    E, T, C = pp.scale_r((E, T, C))

    tc = ()
    for t in T[0]:
        tc = tc + (t,)
        
    for c in C[0]:
        tc = tc + (c,)
    return tc

