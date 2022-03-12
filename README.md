#### 参考文献：
1、A Deep Bayesian Neural Network for Cardiac Arrhythmia Classification with Rejection from ECG Recordings https://arxiv.org/abs/2203.00512

2、Aggregated Residual Transformations for Deep Neural Networks https://ieeexplore.ieee.org/document/8100117

#### 搭建环境
tensorflow  2.5.0

python      3.8.11 

numpy       1.19.1

#### 说明：
Convk块为论文(2)提到的ResNeXt结构，根据论文(1)的参数设置：group=16，kernel_size=16,应该是使用了论文(2)中Fig3(c)结构。

在模型的训练过程中，将强制打开dropout，最后保存模型以及模型的权重，计算模型不确定性时，直接使用该模型进行计算

计算完模型的不确定性后，将会重新创建一个完全相同的模型，并将训练好的模型权重进行加载，对模型确定性高的数据和确定性低的数据分开进行预测
（最后预测结果评价待添加，数据获取函数未添加）