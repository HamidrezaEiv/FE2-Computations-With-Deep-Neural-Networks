import numpy as np
from tensorflow.keras import models, layers, optimizers, regularizers, callbacks
import joblib
import os
from utils import prep
from time import time

def gen_models(act, nn, nl):
    nf = 3
    nv = 9
    reg = regularizers.L2(0.0)
    inp = layers.Input((nf,))
    x = layers.Dense(nn, activation = act, kernel_regularizer = reg)(inp)
    for i in range(nl - 1):
        x = layers.Dense(nn, activation = act, kernel_regularizer = reg)(x)
    out = layers.Dense(nv)(x)
    
    model_c = models.Model(inp, out)
    
    nf = 3
    nv = 3
    reg = regularizers.L2(0.0)
    inp = layers.Input((nf,))
    x = layers.Dense(nn, activation = act, kernel_regularizer = reg)(inp)
    for i in range(nl - 1):
        x = layers.Dense(nn, activation = act, kernel_regularizer = reg)(x)
    out = layers.Dense(nv)(x)
    
    model_t = models.Model(inp, out)
    return (model_c, model_t)

def train_models(test_name, models_, model_name):
    # data
    fdir = '../data/training_data.csv'
    
    if not os.path.exists(f'./{test_name}'):
        os.mkdir(f'./{test_name}')
        
    if not os.path.exists(f'./{test_name}/{model_name}'):
        os.mkdir(f'./{test_name}/{model_name}')
    
    sdir = f'./{test_name}/{model_name}/'
    
    pp = prep(fdir)
    train, val = pp.scale(pp.normscaler)
    
    X, y, y_x = train
    X_v, y_v, y_x_v = val
    
    # training model C
    model = models_[0]
    n_batches = 100
    batch_size = int(len(X) / n_batches)
    
    initial_learning_rate = 1e-3
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000 * n_batches,
        decay_rate=0.1,
        staircase=True)
     
    opt = optimizers.Adam(learning_rate=lr_schedule)
    checkpoint = callbacks.ModelCheckpoint(sdir + 'model_C.h5', monitor='val_loss', verbose=0, mode='min', save_best_only=True, save_weights_only = False)
    
    start_time = time()
    model.compile(optimizer = opt, loss = 'mse')
    hist = model.fit(X, y_x, epochs = 4000, batch_size = batch_size, validation_data = (X_v, y_x_v), callbacks=[checkpoint], verbose = 2)
    end_time = time()
    cp_time = end_time - start_time 
    
    hist = hist.history
    
    hist = np.array(hist)
    np.savez_compressed(sdir + 'res_C', hist = hist, cp_time = cp_time)
    joblib.dump(pp, sdir + 'pp')
    
    # training model T
    model = models_[1]

    initial_learning_rate = 1e-3
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000 * n_batches,
        decay_rate=0.1,
        staircase=True)
     
    opt = optimizers.Adam(learning_rate=lr_schedule)
    checkpoint = callbacks.ModelCheckpoint(sdir + 'model_T.h5', monitor='val_loss', verbose=0, mode='min', save_best_only=True, save_weights_only = False)

    start_time = time()
    model.compile(optimizer = opt, loss = 'mse')
    hist = model.fit(X, y, epochs = 4000, batch_size = batch_size, validation_data = (X_v, y_v), callbacks=[checkpoint], verbose = 2)
    end_time = time()
    cp_time = end_time - start_time 

    hist = hist.history

    hist = np.array(hist)
    np.savez_compressed(sdir + 'res_T', hist = hist, cp_time = cp_time)
