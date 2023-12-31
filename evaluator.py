import torch
from utils import AverageMeter, batch_accuracy
import datetime
import numpy as np


class Evaluator():
    def __init__(self, dataset, data_loader, logger, writer, device, data_type='Test') -> None:
        self.dataset = dataset
        self.device = device
        self.data_loader = data_loader
        self.logger = logger
        self.loss_meters = AverageMeter()
        self.top1_acc = AverageMeter()
        self.top5_acc = AverageMeter()
        self.writer = writer
        self.data_type = data_type

        self.best_acc = -1
        self.best_epoch = -1

    @torch.no_grad()
    def eval(self, model, loss_function, epoch):
        model.eval()
        test_acc = 0.0
        num = 0
        vis_test_label = []
        vis_test_pred = []
        for images, labels, _ in self.data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            pred, loss = self.eval_batch(model, images, labels, loss_function)
            nloss = loss.mean()

            acc = batch_accuracy(pred, labels)
            test_acc += acc
            num += 1
            vis_test_label.append(labels.cpu().numpy())
            vis_test_pred.append(torch.argmax(pred, dim=1).cpu().detach().numpy())

            self.update_meters(pred, labels, nloss)
        self.update_best(epoch)
        self.log(epoch)
        self.write(epoch)
        self.reset_meters()

        test_acc = test_acc / num
        vis_test_label = np.concatenate(vis_test_label)
        vis_test_pred = np.concatenate(vis_test_pred)
        return test_acc, vis_test_label, vis_test_pred

    def eval_batch(self, model, images, labels, loss_function):
        logits = model(images)
        loss = loss_function(logits, labels)
        return logits, loss

    def update_meters(self, pred, labels, loss):
        batch_acc = batch_accuracy(pred, labels)
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.top1_acc.update(batch_acc, labels.shape[0])
        if self.dataset in ['WebVision', 'ILSVRC2012']:
            _, index = pred.topk(5)
            index = index.cpu()
            labels = labels.cpu()
            correct_tmp = index.eq(labels.view(-1, 1).expand_as(index))
            correct_top5 = correct_tmp.sum().cpu().item()
            correct_top5 = float(correct_top5) / float(labels.size(0))
            self.top5_acc.update(correct_top5, labels.shape[0])

    def reset_meters(self):
        self.loss_meters.reset()
        self.top1_acc.reset()
        self.top5_acc.reset()

    def log(self, epoch):
        display = {'dataset': self.dataset,
                   'epoch': epoch,
                   'top1': self.top1_acc.avg,
                   'top5': self.top5_acc.avg,
                   'eval_loss': self.loss_meters.avg,
                   'best_acc': self.best_acc,
                   'best_epoch': self.best_epoch}
        self.logger.info(display)

    def write(self, epoch):
        if self.writer is None:
            return

        self.writer.add_scalar('Accuracy/Eval', self.top1_acc.avg, epoch)
        if self.dataset in ['WebVision', 'ILSVRC2012']:
            self.writer.add_scalar('Accuracy/Eval/top5', self.top5_acc.avg, epoch)
        self.writer.add_scalar('Loss/Eval', self.loss_meters.avg, epoch)

    def update_best(self, epoch):
        if self.top1_acc.avg >= self.best_acc:
            self.best_acc = self.top1_acc.avg
            self.best_epoch = epoch
