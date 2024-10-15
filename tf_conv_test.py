# Based on https://www.tensorflow.org/guide/using_gpu

# module load tensorflow/2.6.0-python-3.8.6

# import tensorflow as tf

input_shape = (1, 362, 370, 251, 64)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv3D(32, 3, input_shape=input_shape[1:], padding='same')(x)
print(y.shape)

# tf.compat.v1.disable_eager_execution()

# print(tf.config.list_physical_devices('GPU'))

# # Creates a graph -- force it to run on the GPU
# input_shape =(4, 28, 28, 28, 1)
# with tf.device('/gpu:0'):
#     x = tf.random.normal(input_shape)
#     y = tf.keras.layers.Conv3D(32, 3, input_shape=input_shape[1:])(x)

# # Creates a session with log_device_placement set to True.
# config=tf.compat.v1.ConfigProto(log_device_placement=False)

# with tf.compat.v1.Session(config=config) as sess:
#   print(sess.run(y))

