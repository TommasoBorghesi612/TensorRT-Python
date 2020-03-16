from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
import time

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Training set setup
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Make batches
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# Make the model
class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.conv2 = Conv2D(64, 4, activation='relu')
        self.conv3 = Conv2D(128, 5, activation='relu')
        self.conv4 = Conv2D(256, 6, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(256, activation='relu')
        self.d3 = Dense(512, activation='relu')
        self.d4 = Dense(10)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x)
        x1 = self.conv3(x)
        x1 = self.conv4(x)
        x2 = self.flatten(x1)
        x3 = self.d1(x2)
        x3 = self.d2(x2)
        x3 = self.d3(x2)
        return self.d4(x3)


# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.\
                 SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.\
                SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    
    predictions = model(images, training=False)
    
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 5


for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    start = time.time()
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    end = time.time()

    inf = end - start
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, \
                Test Accuracy: {}, Inference Time = {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100,
                          inf*1000))


tf.saved_model.save(model, 'testtrt')
