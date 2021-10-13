# 텐서보드를 사용해서 Projector를 구현할 때 오류가 있음
# 이 오류를 해결하기 위해서 작성해야 할 것
from torch.utils.tensorboard import SummaryWriter
import torch
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def set_tensorboard_writer(name):
    writer = SummaryWriter(name) # 'runs/fashion_mnist_experiment_1'
    return writer


def close_tensorboard_writer(writer):
    writer.close()


def show_model_tensorboard_with_no_label(writer, model, train_loader, batch_size):
    # tensorboard에 기록하기
    data_iter = iter(train_loader)
    data = data_iter.next()
    writer.add_graph(model, (data, torch.zeros(1, batch_size, 256)))
