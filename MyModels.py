<<<<<<< HEAD
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')

    def call(self, x):
        # Forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
=======
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')

    def call(self, x):
        # Forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
>>>>>>> e63ff97 (Adding everything)
