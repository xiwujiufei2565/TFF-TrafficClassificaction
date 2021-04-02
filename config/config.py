import os
import numpy as np

class config(object):

    __default_dict__ = {
        "checkpoints_dir": os.path.abspath("./checkpoints"),
        "logs_dir": os.path.abspath("./logs"),
        "config_dir": os.path.abspath("./config"),
        "data_dir": os.path.abspath("./data"),
        "timestep": 60,
        "client_num": 10,   # 客户端的数量
        "c_data_rate": 0.8, # 客户端至少拥有多少数据集的比例
        "num_epochs": 30,
        "shuffle_buffer": 100,
        "batch_size": 64,
        "prefetch_buffer": 10,
    }

    def __init__(self, **kwargs):
        """
        这是参数配置类的初始化函数
        :param kwargs: 参数字典
        """
        self.__dict__.update(self.__default_dict__)
        self.__dict__.update(kwargs)

        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)

    def set(self, **kwargs):
        self.__dict__.update(kwargs)    # 添加配置

    def save_config(self, time):
        """
        保存参数配置类的相关参数
        :param time: 时间点字符串
        :return:
        """
        # 更新相关目录
        self.checkpoints_dir = os.path.join(self.checkpoints_dir, time)
        self.logs_dir = os.path.join(self.logs_dir, time)
        self.config_dir = os.path.join(self.config_dir, time)

        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)
        if not os.path.exists(self.config_dir):
            os.mkdir(self.config_dir)

        config_txt_path = os.path.join(self.config_dir, "config.txt")
        with open(config_txt_path, 'a') as f:
            for key, value in self.__dict__.items():
                if key in ['checkpoints_dir', 'logs_dir', 'config_dir']:
                    value = os.path.join(value, time)
                    s = key + ": " + value + "\n"
                    f.write(s)