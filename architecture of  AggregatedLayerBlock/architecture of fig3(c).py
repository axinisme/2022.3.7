from tensorflow.keras import layers
import tensorflow as tf

DROPOUT_RATE = 0.5


class AggregatedLayer(layers.Layer):

    def __init__(self, dim_num):
        # dim_num的值应为输入总通道数
        # 16为分组数及分组前的通道数，若进行分组，输入的通道数应大于等于分组数
        super(AggregatedLayer, self).__init__()
        self.conv1 = layers.Conv1D(16, 1, activation='swish')
        self.conv2 = layers.Conv1D(16, 16, activation='swish', groups=16, padding="same")
        self.conv3 = layers.Conv1D(dim_num, 1, activation='swish')

    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = layers.Dropout(DROPOUT_RATE)(x)
        x = layers.BatchNormalization()(x)
        x = self.conv2(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        x = layers.BatchNormalization()(x)
        x = self.conv3(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        x = layers.BatchNormalization()(x)
        out = tf.add(inputs, x)
        return out


if __name__ == "__main__":
    inputs = layers.Input([5000, 12])
    out = AggregatedLayer(12)(inputs)
    m = tf.keras.Model(inputs=inputs, outputs=out)
    m.summary()
    tf.keras.utils.plot_model(m, show_shapes=True)
