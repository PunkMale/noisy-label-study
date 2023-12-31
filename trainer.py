import torch
from utils import batch_accuracy
from tqdm import tqdm
import logging
import numpy as np

# keys for batch info
EPOCH = 'epoch'
STEPS = 'steps'
LR = 'batch_lr'
ACC = 'batch_acc'
LOSS = 'batch_loss'


class Trainer:
    def __init__(self, train_dataloader, logger, writer,
                 device, grad_bound=5.0):
        # setting trainer
        self.device = device
        self.train_dataloader = train_dataloader
        self.grad_bound = grad_bound
        self.logger = logger
        self.writer = writer

        # setup log info
        self.setup_info()

    def train(self, model, optimizer, loss_function, epoch):
        model.train()
        self.info[EPOCH] = epoch

        train_acc = 0.0
        num = 0
        vis_label = []
        vis_ground_truth = []
        vis_pred_label = []
        vis_label_confidence = []
        vis_ground_truth_confidence = []
        vis_loss = []
        # 创建一个包含前缀和后缀信息的tqdm对象
        with tqdm(total=len(self.train_dataloader), desc='Epoch {}'.format(self.info[EPOCH]), postfix={'loss': 0.0}, ncols=100) as pbar:
            for images, labels, ground_truths in self.train_dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits, loss = self.train_batch(model, optimizer, loss_function,
                                                images, labels)

                nloss = loss.mean()
                preditions = torch.softmax(logits, dim=1)

                acc = batch_accuracy(preditions, labels)
                vis_label.append(labels.cpu().numpy())
                vis_ground_truth.append(ground_truths.cpu().numpy())
                vis_pred_label.append(torch.argmax(preditions, dim=1).cpu().numpy())
                vis_label_confidence.append(preditions[range(len(labels)), labels].cpu().detach().numpy())
                vis_ground_truth_confidence.append(preditions[range(len(labels)), ground_truths].cpu().detach().numpy())
                vis_loss.append(loss.cpu().detach().numpy())

                train_acc += acc
                num+=1

                self.update_info(optimizer, logits, labels, nloss)
                self.log()
                self.write()

                # 更新后缀信息
                pbar.set_postfix({
                    'lr': f'{self.info[LR]:.4f}',
                    'acc': f'{self.info[ACC]:.4f}',
                    'loss': f'{self.info[LOSS]:.4f}'
                })
                # 更新进度条
                pbar.update(1)

        train_acc = train_acc / num
        vis_label = np.concatenate(vis_label)
        vis_ground_truth = np.concatenate(vis_ground_truth)
        vis_pred_label = np.concatenate(vis_pred_label)
        vis_label_confidence = np.concatenate(vis_label_confidence)
        vis_ground_truth_confidence = np.concatenate(vis_ground_truth_confidence)
        vis_loss = np.concatenate(vis_loss)
        return train_acc, vis_label, vis_ground_truth, vis_pred_label, vis_label_confidence, vis_ground_truth_confidence, vis_loss

    def train_batch(self, model, optimizer, loss_function, images, labels):
        model.zero_grad()
        optimizer.zero_grad()

        logits = model(images)
        loss = loss_function(logits, labels)

        nloss = loss.mean()
        nloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_bound)
        optimizer.step()

        return logits, loss

    def setup_info(self):
        # dict for batch info
        self.info = {}
        self.info[EPOCH] = 0
        self.info[STEPS] = 0
        self.info[LR] = 0
        self.info[ACC] = 0
        self.info[LOSS] = 0

    def update_info(self, optimizer, logits, labels, loss):
        self.info[STEPS] += 1
        self.info[LR] = optimizer.param_groups[0]['lr']
        self.info[ACC] = batch_accuracy(logits, labels)
        self.info[LOSS] = loss.item() \
            if not isinstance(loss, int) \
            else loss

    def log(self):
        self.logger.info(self.info, print_log=False)

    def write(self):
        if self.writer is None:
            return

        self.writer.add_scalar('Learning_Rate/Train', self.info[LR], self.info[STEPS])
        self.writer.add_scalar('Accuracy/Train', self.info[ACC], self.info[STEPS])
        self.writer.add_scalar('Loss/Train', self.info[LOSS], self.info[STEPS])
