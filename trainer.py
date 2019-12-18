import comet_ml as comet
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

# package utils
import os, sys, re
import argparse
from tqdm.notebook import tqdm
import setproctitle, colorama

# src
from src import utils
from src import models, loss, datasets


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description='Training script for LiteFlowNet')

parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--total_epochs', type=int, default=10000, help="Maximum epoch value")
parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")

parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                    help="Spatial dimension to crop training samples for training")
parser.add_argument("--rgb_max", type=float, default=255.)

parser.add_argument('--weight_decay', '-wd', type=float, default=4e-4, metavar='W', help='weight decay parameter')
parser.add_argument('--bias_decay', '-bd', type=float, default=0, metavar='B', help='bias decay parameter')

parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
parser.add_argument('--no_cuda', action='store_true')

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

parser.add_argument('--validation_frequency', type=int, default=1, help='validate every n epochs')
parser.add_argument('--backup_frequency', type=int, default=25, help='save backup at every n epochs')
parser.add_argument('--render_validation', action='store_true',
                    help='run inference (save flows to file) and every validation_frequency epoch')
parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1],
                    help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')

parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to the pre-trained model (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

# For instance
utils.add_arguments_for_module(parser, models, argument_for_class='model', default='LiteFlowNet',
                               parameter_defaults={'starting_scale': 10.0,
                                                   'lowest_level': 1})

utils.add_arguments_for_module(parser, loss, argument_for_class='loss', default='MultiScale',
                               parameter_defaults={'div_scale': 0.2,
                                                   'startScale': 1,
                                                   'l_weight': [0.001, 0.001, 0.001, 0.001, 0.001, 0.01],
                                                   'norm': 'L2'})

utils.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam',
                               skip_params=['params'])

utils.add_arguments_for_module(parser, torch.optim.lr_scheduler, argument_for_class='lr_scheduler', default='MultiStepLR',
                               skip_params=['optimizer'],
                               parameter_defaults={'milestones': [-1]})

utils.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='PIVData',
                               skip_params=['is_cropped', 'transform'],
                               parameter_defaults={'root': './data/piv_datasets',
                                                   'mode': 'train'})

utils.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='PIVData',
                               skip_params=['is_cropped', 'transform'],
                               parameter_defaults={'root': './data/piv_datasets',
                                                   'replicates': 1,
                                                   'mode': 'val'})

utils.add_arguments_for_module(parser, comet, argument_for_class='logger', default='Experiment',
                               exception=['log', 'display'],
                               parameter_defaults={'api_key': '1zB8P6u9ztqAuzy88PWhpbaIU',
                                                   'project_name': 'piv-flownet',
                                                   'workspace': 'flow-diagnostics-itb',
                                                   'parse_args': False})

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)


# Reusable class for training and validation
class Train:
    def __init__(self, args, logger, data_loader, model_and_loss, optimizer, lr_scheduler=None):
        self.args = args
        self.experiment = logger

        self.data_loader = data_loader
        self.model_and_loss = model_and_loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_label = list(model_and_loss.module.loss.loss_labels)[0][0]

    def perform_epoch(self, loader_key, epoch, offset):
        total_epoch_loss = 0
        l_dataloader = len(self.data_loader[loader_key])

        if bool(re.search('val', loader_key)):
            self.model_and_loss.eval()
            title = 'Validating Epoch {}'.format(epoch)
            progress = tqdm(utils.IteratorTimer(self.data_loader[loader_key]), ncols=100, unit='batch',
                            total=l_dataloader, leave=False, position=offset, desc=title)
        elif bool(re.search('train', loader_key)):
            self.model_and_loss.train()
            title = 'Training Epoch {}'.format(epoch)
            progress = tqdm(utils.IteratorTimer(self.data_loader[loader_key]), ncols=120, unit='batch',
                            total=l_dataloader, smoothing=.9, miniters=1, leave=False, position=offset, desc=title)
        else:
            raise ValueError(f'Unknown loader key ({loader_key})! Must contain either "train" or "val" ')

        # Start batch iteration
        for batch_idx, (data, target) in enumerate(progress):
            if self.args.cuda and self.args.number_gpus > 0:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]

            with torch.set_grad_enabled(bool(re.search("train", loader_key))):
                self.optimizer.zero_grad() if bool(re.search('val', loader_key)) else None

                losses = self.model_and_loss(data, target[0])
                batch_loss = losses[0]  # Collect the first loss (MultiScale-{norm})!

                if bool(re.search("train", loader_key)):
                    batch_loss.backward()
                    self.optimizer.step()

                batch_loss_array = batch_loss.item()

                # LOGGER
                log_name = ('_').join([loader_key, 'batch', self.loss_label])
                step_count = (epoch - 1) * l_dataloader + (batch_idx + 1)
                self.experiment.log_metric(log_name, batch_loss_array, step=step_count, epoch=epoch)

                total_epoch_loss += batch_loss_array
                assert not np.isnan(total_epoch_loss)

        epoch_loss = total_epoch_loss / float(l_dataloader)
        progress.close()
        return epoch_loss

    def save_model(self, epoch: int, loss_val: float, offset: int, is_best: bool, filename: str = None, **kwargs
                   ) -> None:

        checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
        param = {'arch': self.args.model,
                 'opt': self.args.optimizer,
                 'model_state_dict': self.model_and_loss.module.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'epoch': epoch,
                 'best_EPE': loss_val,
                 'exp_key': self.experiment.get_key()}

        if self.lr_scheduler is not None:
            sch = {'scheduler': self.args.lr_scheduler,
                   'lr_state_dict': self.lr_scheduler.state_dict()}
            param.update(sch)

        param.update(kwargs)  # for extra input arguments
        utils.save_checkpoint(param, is_best, self.args.save, self.args.model, filename=filename)
        checkpoint_progress.update(1)
        checkpoint_progress.close()

    def __call__(self, **kwargs) -> None:
        progress = tqdm(list(range(self.args.start_epoch, self.args.total_epochs + 1)), miniters=1, ncols=100,
                        unit='epoch', desc='Overall Progress', leave=True, position=0)
        OFFSET = 1
        best_err = args.best_err
        best_epoch = self.args.start_epoch

        for epoch in progress:
            self.experiment.log_current_epoch(epoch)

            for key in self.data_loader.keys():
                if bool(re.search('train', key)):  # Training
                    loss = self.perform_epoch(loader_key=key, epoch=epoch, offset=OFFSET)
                    OFFSET += 1

                elif bool(re.search('val', key)) and ((epoch - 1) % self.args.validation_frequency) == 0:  # Validation
                    loss = self.perform_epoch(loader_key=key, epoch=epoch, offset=OFFSET)
                    OFFSET += 1

                    is_best = loss < best_err
                    if is_best:
                        best_err = loss
                        best_epoch = int(epoch)

                    self.save_model(epoch, best_err, OFFSET, is_best, filename=None)
                    OFFSET += 1

                else:
                    raise ValueError(f'Unknown data_loader key is found! unknown_key = {key}')

                # LOGGER
                log_name = ('_').join([key, self.loss_label])
                self.experiment.log_metric(log_name, loss, step=epoch, epoch=epoch)
                self.experiment.log_metric('best_epoch', best_epoch)

            # Epoch update
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.experiment.log_metric('current_lr', self.lr_scheduler.get_lr()[0], step=epoch, epoch=epoch)

            if ((epoch - 1) % self.args.backup_frequency) == 0:
                self.save_model(epoch, best_err, OFFSET, False, filename=f'backup_{epoch}.pth.tar')

        tqdm.write("\n")


if __name__ == '__main__':
    # ------------------------------ DEBUGGING (temp) ------------------------------
    debug_input = [
        'trainer.py', # '--no_cuda',
        '--crop_size', '64', '64',
        '-b', '2',
        '--seed', '69',
        '--name', 'train_trial',
        '--model', 'LiteFlowNet2', '--model_starting_scale', '10', '--model_lowest_level', '2',
        '--optimizer_lr', '4e-5',
        '--loss_startScale', '2', '--loss_l_weight', '0.001', '0.001', '0.001', '0.001', '0.01', '6.25e-4', '--loss_use_mean', 'false',
        '--lr_scheduler', 'MultiStepLR', '--lr_scheduler_milestones', '120', '240', '360', '480', '600', '--lr_scheduler_gamma', '0.5',
        '--training_dataset', 'PIVLMDB', '--training_dataset_root', '../piv_datasets/cai2018/ztest_lmdb/piv_cai2018',
        '--validation_dataset', 'PIVLMDB', '--validation_dataset_root', '../piv_datasets/cai2018/ztest_lmdb/piv_cai2018',
        '--logger_disabled', 'true']

    sys.argv = debug_input  # Uncomment for debugging

    # ------------------------------ PARSING THE INPUT ------------------------------
    # Parse the official arguments
    with utils.TimerBlock("Parsing Arguments") as block:
        log_args = {}
        args = parser.parse_args()

        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults. Also prepare for the parameters logger
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

            if not bool(re.search('logger', argument)):
                log_args[argument] = value

        # --------------- Class Instantiation ---------------
        # Model and Loss
        args.model_class = utils.module_to_dict(models)[args.model]
        args.loss_class = utils.module_to_dict(loss)[args.loss]

        # Optimizer and Learning Rate Scheduler
        args.optimizer_class = utils.module_to_dict(torch.optim)[args.optimizer]
        if args.lr_scheduler is not None:
            args.lr_scheduler_class = utils.module_to_dict(torch.optim.lr_scheduler)[args.lr_scheduler]

        # Dataset
        args.training_dataset_class = utils.module_to_dict(datasets)[args.training_dataset]
        args.validation_dataset_class = utils.module_to_dict(datasets)[args.validation_dataset]

        # Logger
        args.logger = 'ExistingExperiment' if args.resume else args.logger
        args.logger_class = utils.module_to_dict(comet)[args.logger]

        # Misc
        args.save = os.path.join(args.save, args.name)  # Save directory
        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.resume:
            args.log_file = os.path.join(args.save, 'args_resume.txt')
        else:
            args.log_file = os.path.join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with utils.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus

        gpuargs = {'num_workers': args.effective_number_workers,
                   'pin_memory': True,
                   'drop_last': True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        # Load the transformer
        train_transformer, val_transformer = datasets.get_transform(args)

        data_loader = {}
        if os.path.exists(args.training_dataset_root):
            train_dataset = args.training_dataset_class(args, True, transform=train_transformer,
                                                        **utils.kwargs_from_args(args, 'training_dataset'))
            block.log('Training Dataset: {}'.format(args.training_dataset))
            block.log('Training Input: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][0]])))
            block.log(
                'Training Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][1]])))
            data_loader['train'] = DataLoader(train_dataset, batch_size=args.effective_batch_size, shuffle=True,
                                              **gpuargs)

        if os.path.exists(args.validation_dataset_root):
            validation_dataset = args.validation_dataset_class(args, True, transform=val_transformer,
                                                               **utils.kwargs_from_args(args, 'validation_dataset'))
            block.log('Validation Dataset: {}'.format(args.validation_dataset))
            block.log('Validation Input: {}'.format(' '.join([str([d for d in x.size()])
                                                              for x in validation_dataset[0][0]])))
            block.log('Validation Targets: {}'.format(' '.join([str([d for d in x.size()])
                                                                for x in validation_dataset[0][1]])))
            data_loader['val'] = DataLoader(validation_dataset, batch_size=args.effective_batch_size, shuffle=False,
                                            **gpuargs)

    ## Dynamically load model/loss class with params passed in via "--model_[param]=[value]" or "--loss_[param]=[value]"
    with utils.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()

                kwargs = utils.kwargs_from_args(args, 'model')
                self.model = args.model_class(**kwargs)

                if args.pretrained:
                    if os.path.isfile(args.pretrained):
                        self.model.load_state_dict(torch.load(args.pretrained))
                    else:
                        raise ValueError(f"The PRETRAINED file is not found! Fix the file path ({args.pretrained})!")

                kwargs = utils.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(**kwargs)

            def forward(self, data, target, inference=False):
                output = self.model(data[0], data[1])
                loss_values = self.loss(output, target)

                if inference:
                    return loss_values, output
                else:
                    return loss_values


        model_and_loss = ModelAndLoss(args)

        block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
        block.log('Number of parameters: {}'.format(
            sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

        # assing to cuda or wrap with data parallel, model and loss
        if args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed)

        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        if args.resume:  # Resume from checkpoint
            if os.path.isfile(args.resume):
                block.log("Loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch'] + 1

                args.best_err = checkpoint['best_EPE']
                model_and_loss.module.model.load_state_dict(checkpoint['model_state_dict'])
                block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                raise ValueError(f"The RESUME file is not found! Fix the file path ({args.resume})!")
        else:
            args.best_err = 1e8  # Initial best error
            block.log("Random initialization")

    ## Dynamically load the optimizer with parameters passed in via "--optimizer_[param]=[value]" arguments
    with utils.TimerBlock("Initializing {} Optimizer".format(args.optimizer)) as block:
        level2use = list(range(args.model_lowest_level, 6+1))
        def_id = [i for i, level in enumerate(level2use) if level < 4]

        kwargs = utils.kwargs_from_args(args, 'optimizer')
        param_group = [
            {'params': [p for n, p in model_and_loss.named_parameters() if p.requires_grad and n.endswith(".weight")
                        and ("NetE" in n.split('.')[2] and int(n.split('.')[3]) in def_id)],
             'weight_decay': args.weight_decay,
             'lr': 6e-5},
            {'params': [p for n, p in model_and_loss.named_parameters() if p.requires_grad and n.endswith(".weight")
                        and not ("NetE" in n.split('.')[2] and int(n.split('.')[3]) in def_id)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model_and_loss.named_parameters() if p.requires_grad and n.endswith(".bias")
                        and ("NetE" in n.split('.')[2] and int(n.split('.')[3]) in def_id)],
             'weight_decay': args.bias_decay,
             'lr': 6e-5},
            {'params': [p for n, p in model_and_loss.named_parameters() if p.requires_grad and n.endswith(".bias")
                        and not ("NetE" in n.split('.')[2] and int(n.split('.')[3]) in def_id)],
             'weight_decay': args.bias_decay}
        ]
        optimizer = args.optimizer_class(param_group, **kwargs)

        if args.resume:  # Load checkpoint
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for param, default in list(kwargs.items()):
            block.log("{} = {} ({})".format(param, default, type(default)))

    ## Dynamically load the LR scheduler with parameters passed in via "--lr_scheduler_[param]=[value]" arguments
    if args.lr_scheduler is not None:
        with utils.TimerBlock("Initializing {} Learning rate scheduler".format(args.lr_scheduler)) as block:
            kwargs = utils.kwargs_from_args(args, 'lr_scheduler')
            lr_scheduler = args.lr_scheduler_class(optimizer, **kwargs)

            if args.resume:  # Load checkpoint
                lr_scheduler.load_state_dict(checkpoint['lr_state_dict'])

            for param, default in list(kwargs.items()):
                block.log("{} = {} ({})".format(param, default, type(default)))
    else:
        lr_scheduler = None

    ## Dynamically load the Logger with parameters passed in via "--logger_[param]=[value]" arguments
    with utils.TimerBlock("Initializing {} Comet.ml logger".format(args.logger)) as block:
        if args.resume:  # Load checkpoint
            args.logger_previous_experiment = checkpoint['exp_key']

        kwargs = utils.kwargs_from_args(args, 'logger')
        logger = args.logger_class(**kwargs)

        # Init.
        logger.set_name(args.name)
        logger.log_parameters(log_args)

        for param, default in list(kwargs.items()):
            block.log("{} = {} ({})".format(param, default, type(default)))

    ## Log all arguments to file
    if not args.resume and os.path.isfile(args.log_file):  # Overwrite file!
        os.remove(args.log_file)

    for argument, value in sorted(vars(args).items()):
        block.log2file(args.log_file, '{}: {}'.format(argument, value))
    block.log2file(args.log_file, '------------------- RESUME -------------------\n') if args.resume else None

    # |------------------------------------------------------------------------|
    # |------------------------------ START HERE ------------------------------|
    # |------------------------------------------------------------------------|
    trainer = Train(args, logger, data_loader, model_and_loss, optimizer, lr_scheduler=lr_scheduler)
    trainer()
