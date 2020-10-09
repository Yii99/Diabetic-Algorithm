import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv0 = Conv2D(4, 3, strides=1, activation='relu')
        self.Maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self._block0 = MyBlock(8)
        self._block1 = MyBlock(16)
        self._block2 = MyBlock(32)
        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.dense0 = Dense(8, activation='relu')
        self.dense1 = Dense(2)

    def call(self, inputs, training=False):
        output = self.conv0(inputs)
        output = self.Maxpool(output)
        output = self._block0(output, training=training)
        output = self._block1(output, training=training)
        output = self._block2(output, training=training)
        feature_map = output
        output = self.flatten(output)
        output = self.dropout(output, training=training)
        output = self.dense0(output)
        output = self.dense1(output)
        output = tf.nn.softmax(output)
        return output, feature_map

    class MyBlock(tf.keras.layers.Layer):
        def __init__(self, n_out):
            super(MyBlock, self).__init__()
            self.conv0 = Conv2D(n_out, 1, strides=1, activation='relu')
            self.batch_norm0 = tf.keras.layers.BatchNormalization()
            self.conv1 = Conv2D(n_out, 1, strides=1, activation='relu')
            self.batch_norm1 = tf.keras.layers.BatchNormalization()
            self.n_out = n_out

        def call(self, input, training=False):
            n_in = input.get_shape()[-1]
            h = self.conv0(input)
            h = self.batch_norm0(h, training=training)
            h = self.conv1(h)
            h = self.batch_norm0(h, training=training)
            n_out = self.n_out
            if n_in != n_out:
                shortcut = self.conv0(input)
                shortcut = self.batch_norm0(shortcut, training=training)

            else:
                shortcut = input
            return tf.nn.relu(shortcut + h)