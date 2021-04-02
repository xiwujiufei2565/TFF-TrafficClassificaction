import numpy as np
import collections
import tensorflow as tf
import tensorflow_federated as tff
import os
import matplotlib.pyplot as plt

np.random.seed(0)

# 获取tff联邦学习中自带的Mnist数据集
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# 获取某一客户端id的数据
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])

# 模拟展示该客户用户的手写数据样本
# figure = plt.figure(figsize=(20, 4))
# j = 0
#
# for example in example_dataset.take(40):
#     plt.subplot(4, 10, j + 1)
#     plt.imshow(example['pixels'].numpy(), cmap='gray', aspect='equal')
#     plt.axis('off')
#     j += 1
# plt.show()

NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

# TensorBoard相关配置
logs_root = "./logs"
logs_dir = os.path.join(os.path.abspath(logs_root), "training")
print(logs_dir)
summary_writer = tf.summary.create_file_writer(logs_dir)

def preprocess(dataset):
    # 定义输入和输出
    def batch_format_fn(element):
        return collections.OrderedDict(
            x = tf.reshape(element['pixels'], [-1, 784]),
            y = tf.reshape(element['label'], [-1, 1])
        )

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)
print(example_dataset)
print(list(example_dataset.as_numpy_iterator()))
preprocessed_example_dataset = preprocess(example_dataset)
print(preprocessed_example_dataset.element_spec)

# 构建用户数据集
def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids
    ]

# 模拟客户端
sample_clients = emnist_train.client_ids[0: NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))

# 在tf中创建测试模型
def create_keras_model():

    model_input = tf.keras.layers.Input(shape=(784,))
    x = tf.keras.layers.Dense(10, kernel_initializer='zeros')(model_input)
    x = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs=model_input, outputs=x)

    # return tf.keras.models.Sequential([
    #     tf.keras.layers.Input(shape=(784,)),
    #     tf.keras.layers.Dense(10, kernel_initializer='zeros'),
    #     tf.keras.layers.Softmax(),
    # ])

# 在TFF包装定义的模型
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# 定义联邦学习训练过程
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

# print(str(iterative_process.initialize.type_signature))

state = iterative_process.initialize()

# 开始训练
NUM_ROUNDS = 11
with summary_writer.as_default():
    for round_num in range(2, NUM_ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data)
        for name, value in metrics['train'].items():
            tf.summary.scalar(name, value, step=round_num)
        print('round {:2d}, metrics={}'.format(round_num, metrics))