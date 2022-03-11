import tensorflow as tf
from tensorflow.keras import layers

DROPOUT_RATE = 0.5


class AggregatedLayer(layers.Layer):

    def __init__(self, dim_num, training):
        # dim_num的值应为输入总通道数
        # 16为分组数及分组前的通道数，若进行分组，输入的通道数应大于等于分组数
        super(AggregatedLayer, self).__init__()
        self.conv1 = layers.Conv1D(16, 1, activation='swish')
        self.conv2 = layers.Conv1D(16, 16, activation='swish', groups=16, padding="same")
        self.conv3 = layers.Conv1D(dim_num, 1, activation='swish')
        self.training = training

    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = layers.Dropout(DROPOUT_RATE)(x, training=self.training)
        x = layers.BatchNormalization()(x)
        x = self.conv2(x)
        x = layers.Dropout(DROPOUT_RATE)(x, training=self.training)
        x = layers.BatchNormalization()(x)
        x = self.conv3(x)
        x = layers.Dropout(DROPOUT_RATE)(x, training=self.training)
        x = layers.BatchNormalization()(x)
        out = tf.add(inputs, x)
        return out


class DownSampleLayerBlock(layers.Layer):
    # stage块中包含下采样的部分，dim_num为输入数据的通道数，经过该层，通道数保持不变，但采样数减半
    def __init__(self, dim_num, training):
        super(DownSampleLayerBlock, self).__init__()
        self.conv1 = layers.Conv1D(dim_num, 2, strides=2, activation='swish')  # 进行两倍降采样的参数
        self.conv2 = layers.Conv1D(dim_num, 1, activation='swish')
        self.convk = AggregatedLayer(dim_num)
        self.training = training

    def __call__(self, inputs):
        shortcut_out = layers.MaxPool1D()(inputs)
        x = self.conv1(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(DROPOUT_RATE)(x, training=self.training)
        x = self.convk(x)
        x = self.conv2(x)
        x = layers.Dropout(DROPOUT_RATE)(x, training=self.training)
        x = layers.BatchNormalization()(x)
        out = tf.add(shortcut_out, x)
        return out


class FormerLayerBlock(layers.Layer):
    # stage中不含下采样的部分，target_num为通道扩展的目标值，经过该层，采样点不变，但通道数将会变为target_num
    def __init__(self, target_dim, training):
        super(FormerLayerBlock, self).__init__()
        self.conv1 = layers.Conv1D(target_dim, 2, padding='same', activation='swish')
        self.conv2 = layers.Conv1D(target_dim, 1, activation='swish')
        self.convk = AggregatedLayer(target_dim)
        self.shortcut_conv = layers.Conv1D(target_dim, 1, activation='swish')
        self.training = training

    def __call__(self, inputs):
        shortcut_out = self.shortcut_conv(inputs)
        x = self.conv1(inputs)
        x = layers.Dropout(DROPOUT_RATE)(x, training=self.training)
        x = layers.BatchNormalization()(x)
        x = self.convk(x)
        x = self.conv2(x)
        x = layers.Dropout(DROPOUT_RATE)(x, training=self.training)
        x = layers.BatchNormalization()(x)
        out = tf.add(shortcut_out, x)
        return out


def creat_model(training):
    inputs = layers.Input([5000, 12])  # 输入数据尺寸调整
    # stage1构建
    stage1_out = DownSampleLayerBlock(12, training)(inputs)
    stage1_out = FormerLayerBlock(64, training)(stage1_out)
    print("stage1_out:", stage1_out)
    # stage2构建
    stage2_out = DownSampleLayerBlock(64, training)(stage1_out)
    stage2_out = FormerLayerBlock(160, training)(stage2_out)
    print("stage2_out:", stage2_out)
    # stage3构建
    stage3_out = DownSampleLayerBlock(160, training)(stage2_out)
    stage3_out = FormerLayerBlock(160, training)(stage3_out)
    print("stage3_out:", stage3_out)
    # stage4构建
    stage4_out = DownSampleLayerBlock(160, training)(stage3_out)
    stage4_out = FormerLayerBlock(400, training)(stage4_out)
    stage4_out = FormerLayerBlock(400, training)(stage4_out)
    print("stage4_out:", stage4_out)
    # stage5构建
    stage5_out = DownSampleLayerBlock(400, training)(stage4_out)
    stage5_out = FormerLayerBlock(400, training)(stage5_out)
    stage5_out = FormerLayerBlock(400, training)(stage5_out)
    print("stage5_out:", stage5_out)
    # stage6构建
    stage6_out = DownSampleLayerBlock(400, training)(stage5_out)
    stage6_out = FormerLayerBlock(1024, training)(stage6_out)
    stage6_out = FormerLayerBlock(1024, training)(stage6_out)
    stage6_out = FormerLayerBlock(1024, training)(stage6_out)
    print("stage6_out:", stage6_out)
    # stage7构建
    stage7_out = DownSampleLayerBlock(1024, training)(stage6_out)
    stage7_out = FormerLayerBlock(1024, training)(stage7_out)
    stage7_out = FormerLayerBlock(1024, training)(stage7_out)
    stage7_out = FormerLayerBlock(1024, training)(stage7_out)
    print("stage7_out:", stage7_out)
    # 输出构建
    out = layers.GlobalAvgPool1D()(stage7_out)
    print("pool_out:", out)
    out = layers.Dense(9)(out)  # 这里不使用激活函数是之后会使用未激活的数据
    print("class_out:", out)
    m = tf.keras.Model(inputs=inputs, outputs=out)
    return m


if __name__ == '__main__':
    m = creat_model(training=True)
    # m = tf.keras.Model(inputs=inputs, outputs=out)
    m.summary()
    tf.keras.utils.plot_model(m, show_shapes=True, expand_nested=True)
    m.save('model.h5')
