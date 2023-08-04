import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, regularizers, callbacks, losses
from CustomModel import CustomModel

import joblib
import os
from utils import prep
from time import time

SEED = 24
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')

def gen_models(act, nn, nl):
    nf = 3
    nv = 3
    reg = regularizers.L2(0.0)
    inp = layers.Input((nf,))
    x = layers.Dense(nn, activation = act, kernel_regularizer = reg)(inp)
    for i in range(nl - 1):
        x = layers.Dense(nn, activation = act, kernel_regularizer = reg)(x)
    out = layers.Dense(nv)(x)
    
    model = models.Model(inp, out)

    return model

def train_models(test_name, model, model_name):
    # data
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
    
    # training model
    model = CustomModel(model, [1, 100])
    n_batches = 100
    batch_size = int(len(X) / n_batches)
    
    initial_learning_rate = 1e-3
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000 * n_batches,
        decay_rate=0.1,
        staircase=True)
     
    opt = optimizers.Adam(learning_rate=lr_schedule)
    checkpoint = callbacks.ModelCheckpoint(sdir + 'model', monitor='val_loss_y', 
                                           verbose=0, mode='min', save_best_only=True, 
                                           save_weights_only = False, save_format = 'tf')
    loss = losses.MeanSquaredError()
    start_time = time()
    model.compile(optimizer = opt, loss = loss)
    hist = model.fit(X, [y, y_x], epochs = 4000, batch_size = batch_size, validation_data = (X_v, [y_v, y_x_v]), callbacks=[checkpoint], verbose = 2)
    end_time = time()
    cp_time = end_time - start_time 
    
    # model.save(sdir + 'model')
    hist = hist.history
    
    hist = np.array(hist)
    np.savez_compressed(sdir + 'res', hist = hist, cp_time = cp_time)
    