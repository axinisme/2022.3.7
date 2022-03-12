from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import datetime
import os
import Creat_model

# 未包含数据的获取，仅有模型训练的操作，数据获取后，使用tensorflow构建管道，分别为train_ds和val_ds
# 输入的数据，通道数应在最后，调整输入模型中数据的尺寸，需要调节Creat_model的参数
BATCH_SIZE = 256
epoch = 60000
time = datetime.datetime.now().strftime('%m%d-%H%M')
logdir = 'tensorboard_log/' + time
os.makedirs(logdir)

model = Creat_model.creat_model(training=True)
loss_fn = losses.SparseCategoricalCrossentropy()
opti = optimizers.Adam(learning_rate=0.001)
callback = [callbacks.ReduceLROnPlateau("val_sparse_categorical_crossentropy", 0.3, 6000, verbose=1, mode="max"),
            callbacks.ModelCheckpoint("model.h5", "val_sparse_categorical_crossentropy", verbose=1, save_best_only=1),
            callbacks.TensorBoard(logdir)]

model.compile(optimizer=opti, loss=loss_fn, metrics="sparse_categorical_crossentropy")
model.fit(train_ds, epochs=epoch, validation_data=val_ds, callbacks=callback, verbose=1)

model.save('model_keep_dropout.h5')  # 保存的模型在预测时将始终打开dropout
model.save_weights('model_keep_dropout_weight.h5')
