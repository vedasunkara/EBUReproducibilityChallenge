from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf

class DQNetwork():
  def __init__(self, actions, input_shape,
               minibatch_size=32,
               learning_rate=0.00025,
               discount_factor=0.99,
               dropout_prob=0.1,
               load_path=None,
               logger=None):

      # Parameters
      self.actions = actions 

 

      self.model = Sequential()

      # Second convolutional layer
      self.model.add(Conv2D(64, 4, strides=(3, 3),
                            padding='valid',
                            activation='relu',
                            input_shape=input_shape,
                            data_format='channels_first'))

      # Third convolutional layer
      self.model.add(Conv2D(64, 3, strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            input_shape=input_shape,
                            data_format='channels_first'))

      # Flatten the convolution output
      self.model.add(Flatten())

      # First dense layer
      self.model.add(Dense(512, input_shape=input_shape, activation='relu'))

      # Output layer
      self.model.add(Dense(self.actions))

      # Load the network weights from saved model
      if load_path is not None:
          self.load(load_path)

      self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)#1e-3

      self.model.compile(loss='mean_squared_error',
                         optimizer='rmsprop',
                         metrics=['accuracy'])


  def loss(self,inputs, labels):
           return tf.keras.losses.mean_squared_error(inputs,labels)
