# TensorRT-Python
A first implementation of a TensoRT model with the Python API

The file CNN_model_builder is used to build the CNN Network and to save the frozen model (the model used is overcomplicated on pourpose so when we are going to run the inference time tests we sould be able to see a substantial time save)

The Trt-builder il used to convert the model with TensorRT after the frozen model has been made.

The Trt_inference is used to perform inference with the the Trt_model.

Python Version 2.1

TensorRT Version 6.0

CUDA Version 10.1

CUDNN version 10.1

