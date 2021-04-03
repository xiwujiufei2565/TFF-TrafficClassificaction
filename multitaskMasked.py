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
# time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#
# checkpoint_dir = os.path.join(conf.checkpoints_dir, time)
# if not os.path.exists(checkpoint_dir):
#     os.mkdir(checkpoint_dir)
#
# log_dir = os.path.join(conf.logs_dir, time)
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
#
# conf.save_config(time)
#
# writer = tf.summary.create_file_writer(log_dir)

# 定义预处理过程
def preprocess(dataset):
    # 定义输入和输出
    def batch_format_fn(element):
        return collections.OrderedDict(
            x = tf.reshape(element['input'], [-1, 60, 2]),
            y = [tf.reshape(element['label1'], [-1, 5]), tf.reshape(element['label2'], [-1, 5]), tf.reshape(element['label3'], [-1, 5])]
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

print(federated_train_data[0].element_spec)

# 构建联邦学习测试数据集
federated_test_data = make_federated_data(test_data)

# 构建联邦学习评估数据集
federated_val_data = make_federated_data(val_data)

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

    output1 = Dense(5, activation='softmax', name='Class1')(x)
    output2 = Dense(5, activation='softmax', name='Class2')(x)
    output3 = Dense(5, activation='softmax', name='Class3')(x)

    return Model(inputs=model_input, outputs=[output1, output2, output3])

# 在TFF中定义包装好的模型
def model_fn():
    keras_model = build_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.CategoricalCrossentropy()],
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

# 客户端更新模块的tf代码
@tf.function
def client_update(model, dataset, server_weights, client_optimizer, loss):
    """
    使用server传输的模型参数和client的数据集来训练模型
    :param loss:
    :param model: 客户端模型
    :param dataset: 客户端数据集
    :param server_weights: 服务端参数
    :param client_optimizer: 客户端优化器
    :return: 客户端模型参数
    """
    # 获取客户端模型的参数
    client_weights = model.trainable_variables
    client_loss = loss
    tf.nest.map_structure(lambda x, y: x.assign(y), client_loss, loss)
    # 使用服务端的模型参数来更新客户端的模型参数
    tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)
    loss = tf.keras.losses.CategoricalCrossentropy
    # 使用客户端优化器来对本地模型进行训练
    for batch in dataset:
        with tf.GradientTape() as tape:
            # 计算前向传播
            outputs = model.forward_pass(batch) # -> loss predictions num_examples
        # 计算相应的梯度
        client_loss += outputs.loss
        grads = tape.gradient(outputs.loss, client_weights)
        grads_and_vars = zip(grads, client_weights)

        # 更新梯度
        client_optimizer.apply_gradients(grads_and_vars)
    client_loss /= tf.cast(len(dataset), dtype=tf.float32)
    return client_weights, client_loss

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
    loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    return client_update(model, tf_dataset, server_weights, client_optimizer, loss)

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
    client_weights, clients_loss = tff.federated_map(
        client_update_fn, (federated_dataset, server_weights_at_client)
    )

    # 服务器平均所有客户端更新的模型参数
    mean_client_weights = tff.federated_mean(client_weights)

    # 服务器更新他的模型
    server_weights = tff.federated_map(server_update_fn, mean_client_weights)

    return server_weights, client_weights, clients_loss

# 初始化联邦学习算法
federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn
)

def evaluate(server_state, central_emnist_test):
  keras_model = build_model()
  keras_model.compile(
      loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.CategoricalCrossentropy()],
      metrics=[tf.keras.metrics.CategoricalAccuracy()]
  )

  # 处理数据集
  test_x = np.zeros((1, 60, 2))
  test_y = np.zeros((3, 1, 5))
  for item in central_emnist_test.as_numpy_iterator():
      test_x = np.concatenate((test_x, item['x']), axis=0)
      test_y = np.concatenate((test_y, item['y']), axis=1)
  test_x = test_x[1:,:,:]
  test_y = test_y[:,1:,:]
  test_y1 = test_y[0,:,:]
  test_y2 = test_y[1, :, :]
  test_y3 = test_y[2, :, :]
  test_y1 = test_y1.reshape((-1, 5))
  test_y2 = test_y2.reshape((-1, 5))
  test_y3 = test_y3.reshape((-1, 5))

  keras_model.set_weights(server_state)
  result = keras_model.evaluate(test_x, [test_y1, test_y2, test_y3], verbose=0)
  return result

state = federated_algorithm.initialize()

# 开始训练
# NUM_ROUNDS = 11
# name = ['sum_loss', 'class1_loss', 'class2_loss', 'class3_loss', 'class1_acc', 'class2_acc', 'class3_acc']
# client_name = 'client'
# with writer.as_default():
#     for round_num in range(1, NUM_ROUNDS):
#         state, client_weights, client_loss = federated_algorithm.next(state, federated_train_data)
#         for i in range(conf.client_num):
#             result = evaluate(client_weights[i], federated_train_data[i])
#             for j in range(len(result)):
#                 tf.summary.scalar(str(i) + "_" + client_name + name[j], result[j], step=round_num)
#         # 对每一个客户端进行评估
#         print("NUM_ROUNDS: {}, Loss: {}".format(round_num, np.array(client_loss)))

NUM_ROUNDS = 50
for round_num in range(1, NUM_ROUNDS):
    state, client_weights, client_loss = federated_algorithm.next(state, federated_train_data)
    acc_mean = 0
    for i in range(conf.client_num):
        result = evaluate(state, federated_test_data[i])
        acc_mean += result[6]
    # 对每一个客户端进行评估
    acc_mean /= conf.client_num
    print("NUM_ROUNDS: {}, Acc: {}".format(round_num, acc_mean))

# 使用评估集，对服务端模型进行评估
acc_val_mean = 0
for i in range(conf.client_num):
    result = evaluate(state, federated_val_data[i])
    acc_val_mean = acc_val_mean + result[6]
acc_val_mean = acc_val_mean / conf.client_num
print("Valid Result: ", acc_val_mean)


