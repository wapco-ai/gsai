import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
print("Is Built With CUDA:", tf.test.is_built_with_cuda())
print("Is GPU Available (deprecated):", tf.test.is_gpu_available())