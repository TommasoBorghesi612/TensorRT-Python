import sys
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants

sys.stderr = open('errors.txt', 'w')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

input_data = x_test[..., tf.newaxis]

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(
    max_workspace_size_bytes=(1 << 32))
conversion_params = conversion_params._replace(precision_mode="FP32")
# conversion_params = conversion_params._replace(maximum_cached_engiens=100)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='/home/bot0/Desktop/testtrt',
    conversion_params=conversion_params)
converter.convert()

num_rounds = 10
def my_input_fn():
      for _ in range(num_rounds):
        inp1 = np.random.normal(size=(1, 1, 28 ,28, 1)).astype(np.float32)
        yield inp1


converter.build(input_fn=my_input_fn)
converter.save('testtrt_out3')