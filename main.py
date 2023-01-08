# This is a sample Python script.
import tensorflow
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
from matplotlib import pyplot as plt

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
#
# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print("Tensorflow version " + tf.__version__)

# #try: # detect TPUs
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
#     strategy = tf.distribute.TPUStrategy(tpu)
# except ValueError: # detect GPUs
#     strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines (works on CPU too)
#     #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
#     #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines
#
# # print("Number of accelerators: ", strategy.num_replicas_in_sync)


def display_sinusoid():
  X = range(180)
  Y = [math.sin(x/10.0) for x in X]
  plt.plot(X, Y)
  plt.show()


display_sinusoid()
