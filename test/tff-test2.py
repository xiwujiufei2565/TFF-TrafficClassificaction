import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(10)

# 加载训练数据集
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# -> OrderedDict([('label', TensorSpec(shape=(), dtype=tf.int32, name=None)),
# ('pixels', TensorSpec(shape=(28, 28), dtype=tf.float32, name=None))])

# 处理数据成为输入数据集
NUM_CLIENTS = 10
BATCH_SIZE = 20

def preprocess(dataset):

    def batch_format_fn(element):
        """Flatten a batch of emnist data and return a (features, label) tuple."""
        return (tf.reshape(element['pixels'], [-1, 28*28]),
                tf.reshape(element['label'], [-1, 1]))

    return dataset.batch(BATCH_SIZE).map(batch_format_fn)

# 制作客户数据
clients_ids = np.random.choice(emnist_train.client_ids, size=NUM_CLIENTS, replace=False)
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(i)) for i in clients_ids]

# 创建keras模型
def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zero'),
        tf.keras.layers.Softmax(),
    ])

# 根据keras模型创建tff.learning模型
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# 客户端更新模块的tf代码
@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
    """
    使用server传输的模型参数和client的数据集来训练模型
    :param model: 客户端模型
    :param dataset: 客户端数据集
    :param server_weights: 服务端参数
    :param client_optimizer: 客户端优化器
    :return: 客户端模型参数
    """
    # 获取客户端模型的参数
    client_weights = model.trainable_variables
    # 使用服务端的模型参数来更新客户端的模型参数
    tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)
    # 使用客户端优化器来对本地模型进行训练
    for batch in dataset:
        with tf.GradientTape() as tape:
            # 计算前向传播
            outputs = model.forward_pass(batch)
        # 计算相应的梯度
        grads = tape.gradient(outputs.loss, client_weights)
        grads_and_vars = zip(grads, client_weights)

        # 更新梯度
        client_optimizer.apply_gradients(grads_and_vars)
    return client_weights

# 服务端的更新的tf代码
@tf.function
def server_update(model, mean_client_weights):
    """
    用平均所有客户端的模型参数来更新服务端模型参数
    :param model: 服务端模型
    :param mean_client_weights: 所有客户端平均之后的客户端参数
    :return:
    """
    # 服务端模型参数
    server_weights = model.trainable_variables
    # 更新服务端参数
    tf.nest.map_structure(lambda x, y: x.assign(y), server_weights, mean_client_weights)

    return server_weights

# 初始化服务端模型
@tff.tf_computation
def server_init():
    model = model_fn()
    return model.trainable_variables

# 创建模型参数结构
model_weights_type = server_init.type_signature.result

# 将服务端模型参数定义为联合计算并放置在服务端
@tff.federated_computation
def initialize_fn():
    return tff.federated_value(server_init(), tff.SERVER)

# 创建数据集结构
dummy_model = model_fn()
tf_dataset_type = tff.SequenceType(dummy_model.input_spec)

# 将客户端更新代码转为tff代码
@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
    model = model_fn()
    client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    return client_update(model, tf_dataset, server_weights, client_optimizer)

# 将服务器更新代码转为tff代码
@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
    model = model_fn()
    return server_update(model, mean_client_weights)


# 将数据集结构和模型参数结构转为联邦结构
federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

# 联邦学习过程
@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
    # 将服务器模型广播到客户端上
    server_weights_at_client = tff.federated_broadcast(server_weights)

    # 客户端计算更新过程，并更新参数
    client_weights = tff.federated_map(
        client_update_fn, (federated_dataset, server_weights_at_client)
    )

    # 服务器平均所有客户端更新的模型参数
    mean_client_weights = tff.federated_mean(client_weights)

    # 服务器更新他的模型
    server_weights = tff.federated_map(server_update_fn, mean_client_weights)

    return server_weights, client_weights

# 初始化联邦学习算法
federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn
)

state = federated_algorithm.initialize()

# 开始训练
NUM_ROUNDS = 2
for round_num in range(1, NUM_ROUNDS):
    state, client_weights = federated_algorithm.next(state, federated_train_data)
    print("NUM_ROUNDS: ", round_num)
