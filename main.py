import os
import torch
import argparse
import utils
import random
import numpy as np
from datetime import datetime
from trainer import Trainer
from evaluator import Evaluator
from time import sleep

EXP_ID_LEN = 8
EXP_DIR = 'experiment'
DATA_DIR = 'data'
CONFIG_DIR = 'config'
WRITER_DIR = 'runs'

# get exp id
exp_id = utils.get_exp_id(EXP_ID_LEN)

# argparse
parser = argparse.ArgumentParser(description='Robust Loss with Clean Set')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--config', type=str, default='cifar10_vgg19', required=True)
parser.add_argument('--noise_type', type=str, default='sym')
parser.add_argument('--noise_rate', type=float, default=0.0, required=True)
parser.add_argument('--gpu', action='extend', nargs='+', type=str, required=True)
parser.add_argument('--dataparallel', action='store_true', default=False)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--tuning', action='store_true', default=False)
args = parser.parse_args()

# read config
cfg_path = os.path.join(CONFIG_DIR, args.config + '.json')
model_cfg, loss_cfg, dataset_cfg, optim_cfg = utils.get_config(cfg_path)

# create dir
exp_info = args.config.split('_', 1)
exp_path_dataset = os.path.join(EXP_DIR, exp_info[0])  # mnist/cifar10/  etc.
exp_path_sym = os.path.join(exp_path_dataset, args.noise_type)  # sym/asym
exp_path_model = os.path.join(exp_path_sym, exp_info[1])  # ce/nce/etc.
exp_path_noise = os.path.join(exp_path_model, 'n{}'.format(args.noise_rate))  # n0.0/etc.
for path in [EXP_DIR, DATA_DIR, exp_path_dataset, exp_path_sym, exp_path_model, exp_path_noise]:
    utils.build_dirs(path)  # create dirs


# setup logger
log_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_name = '{}_{}.log'.format(log_time, exp_id)  # log_time_exp_id.log
log_data_path = os.path.join(exp_path_noise, log_name.split('.')[0])
utils.build_dirs(log_data_path)
log_file_path = os.path.join(exp_path_noise, log_name)
logger = utils.setup_logger(name=log_name, log_file=log_file_path)

# setup tensorboard writer
writer = None

# setup device and random seed
logger.info('[Basic Config]')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    logger.info('Using CUDA')
    logger.info('CUDA Version: {}'.format(torch.version.cuda))
    device_list = [torch.cuda.get_device_name(i)
                   for i in range(0, torch.cuda.device_count())]
    logger.info('VISIBLE GPU: %s' % (device_list))
    logger.info('VISIBLE GPU ID: ' + ','.join(args.gpu))
else:
    device = torch.device('cpu')
logger.info('Pytorch Version: {}'.format(torch.__version__))
logger.info('Seed: {}'.format(args.seed))
logger.info('Experiment ID: {}'.format(exp_id))
logger.info('Eval Freq: {}'.format(args.eval_freq))
logger.info('Tuning: {}'.format('True' if args.tuning else 'False'))
logger.info('')


def main():
    # setup model
    model = utils.get_model(model_cfg, dataset_cfg['num_classes'])
    model = model.to(device)
    if args.dataparallel:
        model = torch.nn.DataParallel(model)
    logger.info('[Model Config]')
    for k, v in model_cfg.items():
        logger.info('{}: {}'.format(k, v))
    logger.info('')

    # setup dataset
    dataloaders = utils.get_dataloader(DATA_DIR, dataset_cfg,
                                       args.noise_type, args.noise_rate,
                                       args.tuning)
    train_dataloader, test_dataloader = dataloaders
    # setup trainer and evaluator
    trainer = Trainer(train_dataloader, logger, writer, device, optim_cfg['grad_bound'])
    test_evaluator = Evaluator(dataset_cfg['name'], test_dataloader, logger, writer, device)

    logger.info('[Dataset Config]')
    for k, v in dataset_cfg.items():
        logger.info('{}: {}'.format(k, v))
    logger.info('noise_type: {}'.format(args.noise_type))
    logger.info('noise_rate: {}'.format(args.noise_rate))
    logger.info('num_train_samples: {}'.format(len(train_dataloader.dataset)))
    logger.info('num_test_samples: {}'.format(len(test_dataloader.dataset)))
    logger.info('trans_matrix:')
    for row in train_dataloader.dataset.trans_matrix:
        logger.info(['{:.3f}'.format(col) for col in row])
    logger.info('')

    # setup loss
    loss_function = utils.get_loss(reduction='none')
    loss_function = loss_function.to(device)
    logger.info('[Loss Config]')
    for k, v in loss_cfg.items():
        logger.info('{}: {}'.format(k, v))
    logger.info('')
   
    # setup optim
    optimizer = utils.get_optimizer(optim_cfg['optimizer'],
                                    model.parameters(),
                                    optim_cfg)
    scheduler = utils.get_scheduler(optim_cfg['scheduler'],
                                    optimizer,
                                    optim_cfg)
    logger.info('[Optim Config]')
    for k, v in optim_cfg.items():
        logger.info('{}: {}'.format(k, v))
    logger.info('')

    # start training
    logger.info('[Training]')
    for epoch in range(optim_cfg['total_epoch']):
        # train
        logger.info('=' * 10 + 'Train' + '=' * 10)
        train_acc, vis_label, vis_ground_truth, vis_pred_label, vis_label_confidence, vis_ground_truth_confidence, vis_loss = trainer.train(model, optimizer, loss_function, epoch + 1)
        if optim_cfg['optimizer'] == 'adjust_lr':
            scheduler(optimizer, epoch)
        else:
            scheduler.step()
        # eval
        logger.info('=' * 10 + 'Eval' + '=' * 10)
        test_acc, vis_test_label, vis_test_pred = test_evaluator.eval(model, loss_function, epoch + 1)
        utils.save_data(os.path.join(log_data_path, 'epoch_{}.npy'.format(epoch+1)), epoch+1, train_acc, vis_label, vis_ground_truth, vis_pred_label, vis_label_confidence, vis_ground_truth_confidence, vis_loss, test_acc, vis_test_label, vis_test_pred)
    logger.info('')
    
    sleep(5)  # waiting for tensorboard writting last epoch data


if __name__ == '__main__':
    main()
