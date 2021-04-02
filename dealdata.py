import numpy as np
import os
import tensorflow as tf
import random
import collections
random.seed(10)

from config.config import config

conf = config()

# 根据制定的带宽和延迟标准，将其分成1-5类
def bandwidth_duration_classification(data):

    for i in range(data.shape[0]):
        # 分类带宽
        if data[i, 0] < 10000:
            data[i, 0] = 0
        elif data[i, 0] < 50000:
            data[i, 0] = 1
        elif data[i, 0] < 100000:
            data[i, 0] = 2
        elif data[i, 0] < 1000000:
            data[i, 0] = 3
        else:
            data[i, 0] = 4
        # 分类时延
        if data[i, 1] < 10:
            data[i, 1] = 0
        elif data[i, 1] < 30:
            data[i, 1] = 1
        elif data[i, 1] < 60:
            data[i, 1] = 2
        else:
            data[i, 1] = 3
        # 类型分类
        data[i, 2] = data[i, 2] - 1
    data = data.astype(int)
    return data

# 处理数据集
def dealQUIC():
    timestep = conf.timestep # 使用前k个到达的分组作为输入？
    # 数据集路径
    train_data_dir = os.path.join(conf.data_dir, "trainData.npy")
    train_label_dir = os.path.join(conf.data_dir, "trainLabel.npy")
    test_data_dir = os.path.join(conf.data_dir, "testData.npy")
    test_label_dir = os.path.join(conf.data_dir, "testLabel.npy")
    val_data_dir = os.path.join(conf.data_dir, "valData.npy")
    val_label_dir = os.path.join(conf.data_dir, "valLabel.npy")
    # 加载数据集
    x_train = np.load(train_data_dir)
    y_train = np.load(train_label_dir)
    x_test = np.load(test_data_dir)
    y_test = np.load(test_label_dir)
    x_val = np.load(val_data_dir)
    y_val= np.load(val_label_dir)
    # 处理数据集
    x_train = x_train[:, :timestep * 2]
    y_train = y_train[:, :timestep * 2]
    x_test = x_test[:, :timestep * 2]
    y_test = y_test[:, :timestep * 2]
    x_val = x_val[:, :timestep * 2]
    y_val = y_val[:, :timestep * 2]

    y_train = bandwidth_duration_classification(y_train)
    y_test = bandwidth_duration_classification(y_test)
    y_val = bandwidth_duration_classification(y_val)

    x_train = x_train.reshape((x_train.shape[0], timestep, 2)) # --> x_train.shape = (6139, 60, 2)
    x_test = x_test.reshape((x_test.shape[0], timestep, 2))
    x_val = x_val.reshape((x_val.shape[0], timestep, 2))

    # train_data = []
    #
    # for i in range(conf.client_num):
    #     c_train_x, c_train_y = deal_tf_dataset_for_client(x_train, y_train, i)
    #
    #     # 制作成one-hot编码
    #     c_train_y1 = c_train_y[:, 0]
    #     c_train_y2 = c_train_y[:, 1]
    #     c_train_y3 = c_train_y[:, 2]
    #
    #     c_train_y1 = tf.keras.utils.to_categorical(c_train_y1, num_classes=5).astype(int)
    #     c_train_y2 = tf.keras.utils.to_categorical(c_train_y2, num_classes=5).astype(int)
    #     c_train_y3 = tf.keras.utils.to_categorical(c_train_y3, num_classes=5).astype(int)
    #
    #     # print(c_train_y3)
    #
    #     # 将其制作成为orderedDict序列
    #     train_dict = collections.OrderedDict()
    #     train_dict['input'] = c_train_x
    #     train_dict['label'] = c_train_y3
    #
    #     # train_data.append(train_dict)
    #     train_data.append(tf.data.Dataset.from_tensor_slices(train_dict))

    # 制作总的数据集----这个暂时用不了
    train_data = create_tf_dataset_for_client(x_train, y_train)
    test_data = create_tf_dataset_for_client(x_test, y_test)
    val_data = create_tf_dataset_for_client(x_val, y_val)

    print(train_data)
    print(test_data)
    print(val_data)

    return train_data, test_data, val_data

def create_tf_dataset_for_client(x_data, y_data):
    train_data = []

    for i in range(conf.client_num):
        c_train_x, c_train_y = deal_tf_dataset_for_client(x_data, y_data, i)

        # 制作成one-hot编码
        c_train_y1 = c_train_y[:, 0]
        c_train_y2 = c_train_y[:, 1]
        c_train_y3 = c_train_y[:, 2]

        c_train_y1 = tf.keras.utils.to_categorical(c_train_y1, num_classes=5).astype(int)
        c_train_y2 = tf.keras.utils.to_categorical(c_train_y2, num_classes=5).astype(int)
        c_train_y3 = tf.keras.utils.to_categorical(c_train_y3, num_classes=5).astype(int)

        # print(c_train_y3)

        # 将其制作成为orderedDict序列
        train_dict = collections.OrderedDict()
        train_dict['input'] = c_train_x
        train_dict['label1'] = c_train_y1
        train_dict['label2'] = c_train_y2
        train_dict['label3'] = c_train_y3

        # train_data.append(train_dict)
        train_data.append(tf.data.Dataset.from_tensor_slices(train_dict))
    return train_data

# 创建客户端数据
def deal_tf_dataset_for_client(x_data, y_data, x):
    """
    创建第x个客户端的数据
    :param x_data: 总的数据集
    :param x: 客户端编号
    :return:
    """
    # 制作客户数据集
    client_num = conf.client_num  # 客户端的数量
    data_size = x_data.shape[0]  # 训练数据集的大小
    data_mean_num = int(data_size / client_num) # 每个客户端的平均至多可以拥有的最大数据集大小
    data_base_num = int(data_mean_num * conf.c_data_rate) # 每个客户端的平均至少可以拥有的最大数据集大小
    base_index = x * data_mean_num
    next_index = x * data_mean_num + data_base_num + random.randint(0, data_mean_num - data_base_num)
    return x_data[base_index:next_index], y_data[base_index:next_index]


if __name__ == '__main__':
    dealQUIC()