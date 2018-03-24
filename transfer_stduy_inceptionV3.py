import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3,preprocess_input

from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import keras
def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt=0
    for r,dirs,files in os.walk(directory):
        for dr in dirs:
            cnt+=len(glob.glob(os.path.join(r,dr+"/*")))
    return cnt

#get_nb_files('E:/keras_data/data1/test')
#数据准备
IM_WIDTH,IM_HEIGHT=299,299#InceptionV3指定的图片尺寸
FC_SIZE=1024 # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE=172# 冻结层的数量

train_dir='E:/keras_data/kaggle_cat_dog/train'#训练集数据
test_dir='E:/keras_data/kaggle_cat_dog/test'#测试集数据
nb_classes=2#5
nb_epoch=3
batch_size=16

nb_train_samples=get_nb_files(train_dir)#训练样本个数
nb_classes=len(glob.glob(train_dir+"/*"))#分类数
nb_test_samples=get_nb_files(test_dir)#测试样本个数
nb_epoch=int(nb_epoch)#epcoh数量
batch_size=int(batch_size)
#图片生成器
train_datagen=ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen=ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
#训练数据与测试数据
train_generator=test_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH,IM_HEIGHT),
    batch_size=batch_size,
    class_mode='binary'#categorical
)
test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(IM_WIDTH,IM_HEIGHT),
    batch_size=batch_size,
    class_mode='binary' # categorical
)
# fine-tuning方式一：使用预训练网络的bottleneck特征
#添加新层
def add_new_last_layer(base_model,nb_classes):
    """
    添加最后的层，输入base_model和分类数量；输出新的model
    :param base_model:
    :param nb_classes:
    :return:
    """
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(FC_SIZE,activation='relu')(x) #new FC layer, random init
    predictions=Dense(1,activation='sigmoid')(x) #softmax用于多分类,5层，二分类用sigmoid,用1层
    model=Model(inputs=base_model.input,outputs=predictions)
    return model
#冻上base_model的所有层，这样可以获得好的bottleneck_feature
def setup_to_transfer_learn(model,base_model):
    """Freeze all layers and compile the model
    :param model:
    :param base_model:
    :return:
    """
    for layer in base_model.layers:
        layer.trainable=False
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',#loss='binary'categorical
                  metrics=['accuracy'])
#定义网络框架
base_model=InceptionV3(weights='imagenet',include_top=False)# 预先要下载no_top模型
model=add_new_last_layer(base_model,nb_classes) # 从基本no_top模型上添加新层
setup_to_transfer_learn(model,base_model)# 冻结base_model所有层

#模式1 训练
history_fit=model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    validation_data=test_generator,
    nb_val_samples=nb_test_samples,
    # class_weight='auto'#这里
)
#模型保存
model.save('E:/keras_data/kaggle_cat_dog/transefer_study_model_cat_dog1.h5')

#画图函数
def plot_training(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.plot(epochs,acc,'r.')
    plt.plot(epochs,val_acc,'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs,loss,'r.')
    plt.plot(epochs,val_loss,'r')
    plt.title('Training and validation loss')
    plt.show()
#训练的acc_loss图
plot_training(history_fit)

