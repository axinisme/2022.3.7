from tensorflow.keras import layers
import tensorflow as tf

DROPOUT_RATE = 0.5


class AggregatedLayerBlock(layers.Layer):

    def __init__(self, dim_num):
        # dim_num的值应为总通道数
        super(AggregatedLayerBlock, self).__init__()
        self.conv1 = layers.Conv1D(dim_num, 1, activation='swish')
        self.conv2 = layers.Conv1D(4, 3, activation='swish', padding='same')
        self.conv3 = layers.Conv1D(dim_num, 1, activation='swish', padding='same')

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
        return x


class AggregatedLayer(layers.Layer):
    # dim_num的值应为总通道数
    def __init__(self, dim_num, block_width):
        super(AggregatedLayer, self).__init__()
        self.layer = []
        self.layer_sum_list = []
        for i in range(block_width):
            self.layer.append(AggregatedLayerBlock(dim_num))

    def __call__(self, inputs):
        x = inputs
        for l in self.layer:
            layer_out = l(x)
            self.layer_sum_list.append(layer_out)
        out = tf.add(tf.add_n(self.layer_sum_list), inputs)
        return out