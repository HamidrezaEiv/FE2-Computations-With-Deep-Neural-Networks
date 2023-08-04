# import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, metrics
# import scipy.optimize as sopt

class CustomModel(models.Model):
    def __init__(self, model, coefs, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)
        self.model = model
        self.loss_tracker_1 = metrics.Mean(name="loss_y")
        self.loss_tracker_2 = metrics.Mean(name="loss_yx")
        # self.sopt = lbfgs_optimizer(self.trainable_variables)
        self.flatten = layers.Flatten()
        self.coefs = coefs
        
    def call(self, inputs):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            y = self.model(inputs)
        yx = tape.batch_jacobian(y, inputs)
        yx = self.flatten(yx)
        return y, yx
   
    @tf.function
    def train_step(self, data):
        x, Y = data
        with tf.GradientTape(persistent=True) as tape:
            y_p, yx_p = self(x, training=True)
            loss_y = self.loss(Y[0], y_p)
            loss_yx = self.loss(Y[1], yx_p)
            loss = self.coefs[0] * loss_y + self.coefs[1] * loss_yx
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker_1.update_state(loss_y)
        self.loss_tracker_2.update_state(loss_yx)
        return {"loss_y": self.loss_tracker_1.result(), "loss_yx": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        x, Y = data
        y_p, yx_p = self(x, training=False)
        loss_y = self.loss(Y[0], y_p)
        loss_yx = self.loss(Y[1], yx_p)
        
        self.loss_tracker_1.update_state(loss_y)
        self.loss_tracker_2.update_state(loss_yx)
        return {"loss_y": self.loss_tracker_1.result(), "loss_yx": self.loss_tracker_2.result()}

    
    
# class lbfgs_optimizer():
#     def __init__(self, trainable_vars, method = 'L-BFGS-B'):
#         super(lbfgs_optimizer, self).__init__()
#         self.trainable_variables = trainable_vars
#         self.method = method
        
#         self.shapes = tf.shape_n(self.trainable_variables)
#         self.n_tensors = len(self.shapes)

#         count = 0
#         idx = [] # stitch indices
#         part = [] # partition indices
    
#         for i, shape in enumerate(self.shapes):
#             n = np.product(shape)
#             idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
#             part.extend([i]*n)
#             count += n
    
#         self.part = tf.constant(part)
#         self.idx = idx
    
#     def assign_params(self, params_1d):
#         params_1d = tf.cast(params_1d, dtype = tf.float32)
#         params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
#         for i, (shape, param) in enumerate(zip(self.shapes, params)):
#             self.trainable_variables[i].assign(tf.reshape(param, shape))       
    
#     def minimize(self, func):
#         init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
#         results = sopt.minimize(fun = func, 
#                             x0 = init_params, 
#                             method = self.method,
#                             jac = True, options = {'iprint' : 0,
#                                                    'maxiter': 1,
#                                                    'maxfun' : 1,
#                                                    'maxcor' : 50,
#                                                    'maxls': 50,
#                                                    'gtol': 1.0 * np.finfo(float).eps,
#                                                    'ftol' : 1.0 * np.finfo(float).eps,
#                                                    'disp' : False})
        
# loss_y = tf.reduce_mean(tf.square(Y[0] - y_p))
# loss_yx = tf.reduce_mean(tf.square(Y[1] - yx_p))