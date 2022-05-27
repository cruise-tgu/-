from tensorflow.keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import preprocess
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

# 训练参数
from main import model

batch_size = 128
epochs = 5
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
# 输入卷积的时候还需要修改一下，增加通道数目
x_train, x_valid, x_test = x_train[:,:,np.newaxis], x_valid[:,:,np.newaxis], x_test[:,:,np.newaxis]
# 输入数据的维度
input_shape =x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

# 定义卷积层
def alexnet(filters, kernerl_size, strides, conv_padding, pool_padding,  pool_size, BatchNormal):
    model.add(Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
                     padding=conv_padding, kernel_regularizer=l2(1e-4)))
    if BatchNormal:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, padding=pool_padding,))
    return model


model = Sequential()
# 搭建输入层，第一层卷积。因为要指定input_shape，所以单独放出来
model.add(Conv1D(filters=96, kernel_size=3, strides=1, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2,strides=2))
#第二层卷积
model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2,strides=2))
#第三层卷积
model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(Activation('relu'))
#第四层卷积
model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(Activation('relu'))
#第五层卷积
model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2,strides=2))

model.add(Flatten())

model.add(Dense(units=2048, activation='relu', kernel_regularizer=l2(1e-4)))

model.add(Dense(units=2048, activation='relu', kernel_regularizer=l2(1e-4)))

model.add(Dense(units=10, activation='softmax', kernel_regularizer=l2(1e-4)))



model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard调用查看一下训练情况
tb_cb=TensorBoard(log_dir='logs')

# 开始模型训练
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
          callbacks=[tb_cb])

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失：", score[0])
print("测试集上的损失:",score[1])
plot_model(model=model, to_file='wdcnn.png', show_shapes=True)