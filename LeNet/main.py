import os
import argparse
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn import Accuracy
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor
from mindspore import Model

from dataset import Mnist_data
from model import LeNet5

parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
parser.add_argument('--device_target', type=str, default='CPU', choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)


def train_net(args, model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = Mnist_data.create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)


def test_net(network, model, data_path):
    """定义验证的方法"""
    ds_eval = Mnist_data.create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))

net = LeNet5()

train_epoch = 1
mnist_path = "../datasets/MNIST_Data"
dataset_size = 1
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# 应用模型保存参数
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(args, model, train_epoch, mnist_path, dataset_size, ckpoint, False)
test_net(net, model, mnist_path)