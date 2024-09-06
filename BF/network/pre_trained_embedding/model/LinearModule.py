import numbers
from typing import Union, Sequence

import tensorflow as tf
import numpy as np

def glorot_uniform(shape, dtype=tf.float32):
    return tf.keras.initializers.VarianceScaling(mode='fan_avg', distribution='uniform')(shape, dtype=dtype)

def truncated_normal(shape, std, dtype=tf.float32):
    return tf.keras.initializers.TruncatedNormal(stddev=std)(shape, dtype=dtype)
        
class Linear(tf.Module):

    def output_shape(self):
        return self._output_shape

    def __init__(self,
                 num_output: Union[int, Sequence[int]],
                 num_input_dims: int = 1,
                 num_input = 1,
                 name: str = 'linear',
                 initializer='linear',
                 global_config=None,
                 dtype=tf.float32,
                 load=False):

        super(Linear, self).__init__(name=name)

        with self.name_scope:
            self.DTYPE = dtype
            self.gc = global_config
            if self.gc == None:
                self.gc = {'iter': 0}
            if isinstance(num_output, numbers.Integral):
                self.output_shape = (num_output,)
            else:
                self.output_shape = tuple(num_output)

            self.num_output = num_output
            self.num_input = num_input

            shape = [self.num_input, self.num_output]
            if initializer == 'linear':
                scale = np.sqrt(1./self.num_input)
            elif initializer == 'relu':
                scale = np.sqrt(2./self.num_input)

            if initializer != 'gate' and initializer != 'final':
                self.w = tf.Variable(truncated_normal(shape, scale, dtype=self.DTYPE), name='weights', dtype=self.DTYPE)
                self.b = tf.Variable(tf.zeros(self.output_shape, dtype=self.DTYPE), name='bias', dtype=self.DTYPE)
            else:
                self.w = tf.Variable(tf.zeros(shape, dtype=self.DTYPE), name='weights', dtype=self.DTYPE)
                if initializer != 'final':
                    self.b = tf.Variable(tf.ones(self.output_shape, dtype=self.DTYPE), name='bias', dtype=self.DTYPE) #tf.Variable(1., shape=tf.TensorShape(None), name='bias')
                else:
                    self.b = tf.Variable(tf.zeros(self.output_shape, dtype=self.DTYPE), name='bias', dtype=self.DTYPE)

            self.num_input_dims = num_input_dims
            self.num_output_dims = len(self.output_shape)
            
            
    def __call__(self, inputs, **kwargs):
        output = tf.matmul(inputs, self.w) + self.b
        return output
