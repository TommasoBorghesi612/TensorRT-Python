import tensorflow as tf
import tensorrt as trt
import numpy as np

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

input_data = tf.constant(x_test[..., tf.newaxis], dtype=tf.float32)

saved_model_loaded = tf.saved_model.load(
    'testtrt_out3')
graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)

# Inferenza
output = frozen_func(input_data)[0].numpy()

print(output)
print('ciao')

with open('out.txt', 'w') as f:
    print(output, file=f)