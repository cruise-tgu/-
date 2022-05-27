from datetime import date

from pytest import mark
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
#from keras.regularizers import l2
import preprocess
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

# 训练参数
batch_size = 128
epochs = 8
num_classes = 10
length = 2048
BatchNorm = True # 是否批量归一化
number = 1000 # 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例

path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=rate,
                                                                  enc=True, enc_step=28)

x_train, x_valid, x_test = x_train[:,:,np.newaxis], x_valid[:,:,np.newaxis], x_test[:,:,np.newaxis]

input_shape =x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

model_name = "lstm_diagnosis-20-{}".format(mark)

# 实例化一个Sequential
model = Sequential()

model.add(LSTM(32, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True))


model.add(Flatten())

# 添加全连接层
model.add(Dense(32))
model.add(Activation("relu"))

# 增加输出层，共num_classes个单元
#model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))
model.add(Dense(units=num_classes, activation='softmax'))


# 编译模型
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard调用查看一下训练情况
tb_cb = TensorBoard(log_dir='logs')

# 开始模型训练
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
          callbacks=[tb_cb])

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])
plot_model(model=model, to_file='lstm-diagnosis.png', show_shapes=True)