# 텐서보드를 사용해서 Projector를 구현할 때 오류가 있음
# 이 오류를 해결하기 위해서 작성해야 할 것
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# console: tensorboard --logdir=runs --bind_all
# # nohup tensorboard --logdir=runs --bind_all > /dev/null 2>&1


def set_tensorboard_writer(name):
    writer = SummaryWriter(name) # 'runs/fashion_mnist_experiment_1'
    return writer


def inspect_model(writer, model, data):
    writer.add_graph(model, data)


def close_tensorboard_writer(writer):
    writer.close()


def add_dataset_figure(writer, dataloader, desc="Train", epoch=0):
    dataiter = iter(dataloader)
    waveform01, waveform02, _, _ = dataiter.next()
    fig = plt.figure()
    plt.plot(waveform01[0].t().numpy(), alpha=0.5)
    plt.plot(waveform02[0].t().numpy(), alpha=0.5)
    plt.title("sample waveform visualization")
    writer.add_figure('Visualize{}'.format(desc), fig, epoch)
    plt.close()


def visualization_dataset_by_byol(writer, dataloader, desc="Train", epoch=0):
    dataiter = iter(dataloader)
    waveform01, waveform02 = dataiter.next()
    fig = plt.figure()
    plt.plot(waveform01[0].t().numpy(), alpha=0.5)
    plt.plot(waveform02[0].t().numpy(), alpha=0.5)
    plt.title("sample waveform visualization")
    writer.add_figure('Visualize{}'.format(desc), fig, epoch)
    plt.close()
    writer.add_audio('Visualize{}'.format(desc), waveform01[0], epoch, sample_rate=16000)




def add_dataset_figure_by_byol(writer, dataloader, desc="Train", epoch=0):
    dataiter = iter(dataloader)
    waveform01, waveform02 = dataiter.next()
    fig = plt.figure()
    plt.plot(waveform01[0].t().numpy(), alpha=0.5)
    plt.plot(waveform02[0].t().numpy(), alpha=0.5)
    plt.title("sample waveform visualization")
    writer.add_figure('Visualize{}'.format(desc), fig, epoch)
    plt.close()


def add_latent_heatmap(writer, data, title, desc, epoch):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.matshow(data)
    writer.add_figure('{}/{}'.format(title, desc), fig, epoch)
    plt.close()


def add_byol_latent_heatmap(writer, online1, online2, target1, target2, title, desc, epoch):
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.matshow(online1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.matshow(online2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.matshow(target2)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.matshow(target1)
    writer.add_figure('{}/{}'.format(title, desc), fig, epoch)
    plt.close()





