import tensorflow as tf
import numpy as np
import Creat_model


REPEAT_NUM = 50
UNCERTAINTY_THRESH = 0.5
model_keep_dropout = tf.keras.models.load_model('model_keep_dropout.h5')

# --------分别对单条数据进行预测，并得到模型的不确定性---------
uncertainty_list = []
for data in data_test:
    data = np.expand_dims(data, 0)
    data = np.repeat(data, REPEAT_NUM, 0)
    result = model_keep_dropout.predict(data).numpy()
    # temp变量计算的均为yi的值
    total_uncertainty_temp = np.sum(result, 0) / REPEAT_NUM
    total_uncertainty = -np.sum(total_uncertainty_temp * np.log(total_uncertainty_temp))

    data_uncertainty_temp = result * np.log(result)
    data_uncertainty = -np.sum(np.sum(data_uncertainty_temp, 2)) / REPEAT_NUM
    model_uncertainty = total_uncertainty - data_uncertainty
    uncertainty_list.append(model_uncertainty)
# ------------------

uncertainty_array = np.array(uncertainty_list)
certain_data_number = np.where(uncertainty_array < UNCERTAINTY_THRESH)
uncertain_data_number = np.where(uncertainty_array >= UNCERTAINTY_THRESH)
certain_data = data_test[certain_data_number]
uncertain_data = data_test[uncertain_data_number]

model_off_dropout = Creat_model.creat_model(training=False)
model_off_dropout.load_weights("model_keep_dropout_weight.h5")  # 使用该模型进行最后的预测
certain_class = model_off_dropout.predict(certain_data)
uncertain_class = model_off_dropout.predict(uncertain_data)
