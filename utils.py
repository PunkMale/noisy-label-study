import json
import os
import random
import string
import logging
import datetime
import torch
import numpy
import torch.optim as optim
from torch.utils.data import DataLoader
from loss import *


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt if self.cnt != 0 else 0


def batch_accuracy(outputs, labels):
    _, pred = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (pred == labels).sum().item()
    return correct / total


def log_display(**kwargs):
    display = '|'
    for key, value in kwargs.items():
        display += ' {}:{:5.3f} |'.format(key, value)
    return display


class Logger(object):
    def __init__(self, name, log_file, level=logging.INFO) -> None:
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        """To setup as many loggers as you want"""
        formatter = logging.Formatter('%(asctime)s %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # 移除控制台处理程序
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

    def info(self, msg, print_log=True):
        self.logger.info(msg)
        if print_log:
            print(datetime.datetime.now(), msg)


def setup_logger(name, log_file, level=logging.INFO):
    logger = Logger(name, log_file, level)
    return logger


def get_exp_id(length):
    exp_id = ''
    for _ in range(length):
        choosen_str = random.SystemRandom().choice(string.ascii_uppercase +
                                                   string.digits)
        exp_id += choosen_str
    return exp_id


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def get_config(path):
    with open(path) as json_file:
        cfg = json.load(json_file)

    return cfg['model'], cfg['loss'], cfg['dataset'], cfg['optim']


def get_model(config, num_classes):
    # VGG
    if config['name'] == "vgg19":
        from torchvision.models import vgg19
        model = vgg19(pretrained=False, num_classes=num_classes)
    # Resnet
    elif config['name'] == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained=False, num_classes=num_classes)
    elif config['name'] == 'resnet34':
        from torchvision.models import resnet34
        model = resnet34(pretrained=False, num_classes=num_classes)
    elif config['name'] == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(pretrained=False, num_classes=num_classes)
    elif config['name'] == 'resnet101':
        from torchvision.models import resnet101
        model = resnet101(pretrained=False, num_classes=num_classes)
    elif config['name'] == 'resnet152':
        from torchvision.models import resnet152
        model = resnet152(pretrained=False, num_classes=num_classes)
    # Densenet
    elif config['name'] =='densenet121':
        from torchvision.models import densenet121
        model = densenet121(pretrained=False, num_classes=num_classes)
    # MobileNet
    elif config['name'] =='mobilenetv2':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(pretrained=False, num_classes=num_classes)
    # Error
    else:
        raise ValueError('model name error')

    return model


def get_loss(reduction='mean'):
    loss = ce(reduction)
    return loss


def get_dataloader(root, config, noise_type, noise_rate, tuning):
    name = config['name']
    if name == 'cifar10':
        from dataset import cifar10
        train_dataset, test_dataset = cifar10(root, noise_type, noise_rate, tuning)
    elif name == 'cifar100':
        from dataset import cifar100
        train_dataset, test_dataset = cifar100(root, noise_type, noise_rate, tuning)
    else:
        raise ValueError('dataset name error')

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config['train_batchsize'],
                                  shuffle=True,
                                  num_workers=config['num_workers'])

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config['test_batchsize'],
                                 shuffle=False,
                                 num_workers=config['num_workers'])

    return train_dataloader, test_dataloader


def get_optimizer(name, params, config):
    nesterov = config['nesterov'] \
        if 'nesterov' in config \
        else False
    if name == 'sgd':
        optimizer = optim.SGD(params,
                              lr=config['learning_rate'],
                              momentum=config['momentum'],
                              weight_decay=config['weight_decay'],
                              nesterov=nesterov)
    elif name == 'adam':
        optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    else:
        raise ValueError('optimizer name error')

    return optimizer


def adjust_learning_rate(self, optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = self.alpha_plan[epoch]
        param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1


def get_scheduler(name, optimizer, config):
    if name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=config['T_max'],
                                                         eta_min=config['eta_min'])
    elif name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=config['step_size'],
                                              gamma=config['gamma'])
    elif name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=config['gamma'],
                                                         milestones=config['milestones'], verbose=True)
    else:
        raise ValueError('scheduler name error')

    return scheduler


def save_data(path, epoch, train_acc, vis_label, vis_ground_truth, vis_pred_label, vis_label_confidence, vis_ground_truth_confidence, vis_loss, test_acc, vis_test_label, vis_test_pred):
    data = {
        'epoch': epoch,
        'train_acc': train_acc,
        'vis_label': vis_label,
        'vis_ground_truth': vis_ground_truth,
        'vis_pred_label': vis_pred_label,
        'vis_label_confidence': vis_label_confidence,
        'vis_ground_truth_confidence': vis_ground_truth_confidence,
        'vis_loss': vis_loss,
        'test_acc': test_acc,
        'vis_test_label': vis_test_label,
        'vis_test_pred': vis_test_pred
    }
    numpy.save(path, data)
    print('saved data to {}'.format(path))


def read_data(path):
    data = numpy.load(path, allow_pickle=True).item()
    return data


if __name__ == '__main__':
    path = '/home/fangzh21/code/noisy-label-study/experiment/cifar10/sym/res34/n0.2/2023-12-31-19-12-59_BGE9JOOF/epoch_8.npy'
    data = read_data(path)
    print(data)
