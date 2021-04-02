import os
import numpy as np
import collections
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Activation, Dense
from tensorflow.keras import Model
import tensorflow_federated as tff
import datetime
from config.config import config
from dealdata import dealQUIC

conf = config()

# 初始化相关日志文件和TensorBoard
time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

checkpoint_dir = os.path.join(conf.checkpoints_dir, time)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

log_dir = os.path.join(conf.logs_dir, time)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

conf.save_config(time)

writer = tf.summary.create_file_writer(log_dir)

# input_spec = collections.OrderedDict(
#     x = tf.TensorSpec(shape=[None, 60, 2], dtype=tf.float32),
#     y = [
#         tf.TensorSpec(shape=[None, 5], dtype=tf.int32),
#         tf.TensorSpec(shape=[None, 4], dtype=tf.int32),
#         tf.TensorSpec(shape=[None, 5], dtype=tf.int32)
#     ]
# )

# 定义预处理过程
def preprocess(dataset):
    # 定义输入和输出
    def batch_format_fn(element):
        return collections.OrderedDict(
            x = tf.reshape(element['input'], [-1, 60, 2]),
            y = tf.reshape(element['label'], [-1, 5])
        )

    return dataset.repeat(conf.num_epochs).batch(conf.batch_size).shuffle(conf.batch_size).map(batch_format_fn)

# 构建用户数据集
def make_federated_data(client_data):
    return [
        preprocess(client_data[x]) for x in range(conf.client_num)
    ]

# 获取数据
train_data, test_data, val_data = dealQUIC()

# 构建联邦学习数据集
federated_train_data = make_federated_data(train_data)

# 构建联邦学习测试数据集
federated_test_data = make_federated_data(test_data)

# 构建模型
def build_model():
    model_input = Input(shape=(conf.timestep, 2))

    x = Conv1D(32, 3, activation='relu', name='conv1')(model_input)
    x = Conv1D(32, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(64, 3, activation='relu', name='conv2')(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(128, 3, activation='relu', name='conv3')(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)

    output = Dense(5, activation='softmax', name='Class')(x)

    return Model(inputs=model_input, outputs=output)

# 在TFF中定义包装好的模型
def model_fn():
    keras_model = build_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

if __name__ == "__main__":

    # Debug
    # print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
    # print('First dataset: {d}'.format(d=federated_train_data))

    # 定义联邦学习训练过程
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    state = iterative_process.initialize()

    NUM_ROUNDS = 51

    with writer.as_default():
        for round_num in range(1, NUM_ROUNDS):
            state, metrics = iterative_process.next(state, federated_train_data)
            for name, value in metrics['train'].items():
                tf.summary.scalar(name, value, step=round_num)
            print('round {:2d}, metrics={}'.format(round_num, metrics))

    evaluation = tff.learning.build_federated_evaluation(model_fn=model_fn)
    train_metrics = evaluation(state.model, federated_test_data)
    print(train_metrics)
