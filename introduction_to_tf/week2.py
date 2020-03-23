import tensorflow as tf
from os import path, getcwd, chdir
from tensorflow import keras
print(tf.__version__)

class EarlyStopping(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if (logs.get('accuracy') > 0.99):
      self.model.stop_training = True

#Minst DAten aus Keras laden und normalisieren auf einen Wert zw. 0 und 1 
mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images  = training_images / 255.0
test_images = test_images / 255.0 

print(training_images[0])

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
                                    
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, callbacks=EarlyStopping()) 
