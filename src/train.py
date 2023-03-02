import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import flow_transforms
import models
import datasets
from loss import compute_semantic_pos_loss
from tensorboardX import SummaryWriter
import datetime
from train_util import *

'''
Main code for training 

author: Fengting 
Last modification: March 8th, 2019
'''

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch SpixelFCN Training on BSDS500',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ================ data setting ====================
    parser.add_argument('--dataset', metavar='DATASET', default='BSD500', choices=dataset_names,
                        help='dataset type : ' + ' | '.join(dataset_names))
    parser.add_argument('--arch', '-a', metavar='ARCH', default='SpixelNet1l_bn',
                        help='model architecture')
    parser.add_argument('--dump_root', metavar='DIR', default='./dump_root',
                        help='path to input dataset')
    parser.add_argument('--output_dir', default='./ckpt_log',
                        help='path to save ckpt')

    parser.add_argument('--train_img_h', '-t_img_h', default=208, type=int, help='img height')
    parser.add_argument('--train_img_w', '-t_img_w', default=208, type=int, help='img width')
    parser.add_argument('--val_img_h', '-v_img_h', default=320, type=int, help='img height_must be 16*n')  #
    parser.add_argument('--val_img_w', '-v_img_w', default=320, type=int, help='img width must be 16*n')

    # ======== training setting ================
    parser.add_argument('--epoch_end', default=300000, type=int, metavar='N', help='number of total epochs, make it big enough to follow the iteration maximum')
    parser.add_argument('--epoch_start', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--epoch_size', default=6000, help='choose any value > 408 to use all the train and val data')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameter for adam')
    parser.add_argument('--weight_decay', '--wd', default=4e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--bias_decay', default=0, type=float, metavar='B', help='bias decay, we never use it')
    parser.add_argument('--additional_step', default=100000, help='the additional iteration, after lr decay')
    parser.add_argument('--down_size', default=16, type=float, help='grid cell size for superpixel training ')
    parser.add_argument('--pos_weight', '-p_w', default=0.003, type=float, help='weight of the pos term')
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu id')
    parser.add_argument('--pretrained_model', default='./pretrained_ckpt/SpixelNet_bsd_ckpt.tar', help='path to pretrained model')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--milestones', default=[200000], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
    parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'], help='optimizer algorithms, we use adam')

    # ================= record setting ===================
    parser.add_argument('--record_dir', default='./training_record', type=str, help='dir to store the tensorboardX writer')
    parser.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (step)')
    parser.add_argument('--record_freq', '-rf', default=5, type=int, help='record frequency (epoch)')
    parser.add_argument('--label_factor', default=5, type=int, help='constant multiplied to label index for viz.')
    parser.add_argument('--no_date', action='store_true', help='don\'t append date timestamp to folder')

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    best_EPE = -1
    n_iter = 0

    output_dir = os.path.abspath(args.output_dir) + '/' + os.path.join(args.dataset)
    train_writer = SummaryWriter(os.path.join(args.record_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(args.record_dir, 'val'))

    # ========== preprocess data ============================
    train_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    val_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    label_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
    ])

    co_transform = flow_transforms.Compose([
        flow_transforms.RandomCrop((args.train_img_h, args.train_img_w)),
        flow_transforms.RandomVerticalFlip(),
        flow_transforms.RandomHorizontalFlip()
    ])

    # =============== load data ==================================================
    print("=> load img {} from '{}'".format(args.dataset, args.dump_root))
    train_set, val_set = datasets.__dict__[args.dataset](
        args.dump_root,
        train_transform=train_transform,
        val_transform=val_transform,
        label_transform=label_transform,
        co_transform=co_transform
    )
    print('The {} train samples and {} val samples found.'.format(len(train_set), len(val_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True)

    # ============== create model =================================================
    if args.pretrained_model:
        network = torch.load(args.pretrained_model)
        args.arch = network['arch']
        print("=> use pretrained model '{}'".format(args.arch))
    else:
        network = None
        print("=> create model '{}'".format(args.arch))

    model = models.__dict__[args.arch](data=network).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # =========== create optimizer, we use adam by default ==================
    assert (args.solver in ['adam', 'sgd'])
    print('=> setting {} as optimizer'.format(args.solver))
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    # for continue training
    if args.pretrained_model and ('dataset' in network):
        if args.pretrained_model and args.dataset_name == network['dataset']:
            optimizer.load_state_dict(network['optimizer'])
            best_EPE = network['best_EPE']
            args.start_epoch = network['epoch']

    print('=> save to save_path: {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # spixelID: superpixel ID for visualization,
    # XY_feat: the coordinate feature for position loss term
    train_spixelID, train_XY_feat = init_spixel_grid(args, b_train=True)
    val_spixelID, val_XY_feat = init_spixel_grid(args, b_train=False)

    #  (0, 300000)
    for epoch in range(args.epoch_begin, args.epoch_end):
        # train for one epoch
        train_avg_slic, train_avg_sem, iteration = train(train_loader, model, optimizer, epoch, train_writer,
                                                         train_spixelID, train_XY_feat)
        if epoch % args.record_freq == 0:
            train_writer.add_scalar('train_avg_slic', train_avg_slic, epoch)
            train_writer.add_scalar('train_avg_sem', train_avg_sem, epoch)

        # evaluate on validation set and save the module( and choose the best)
        with torch.no_grad():
            val_avg_slic, val_avg_sem = validate(val_loader, model, epoch, val_writer, val_spixelID, val_XY_feat)
            if epoch % args.record_freq == 0:
                val_writer.add_scalar('val_avg_slic', val_avg_slic, epoch)
                val_writer.add_scalar('val_avg_sem', val_avg_sem, epoch)

        rec_dict = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'optimizer': optimizer.state_dict(),
            'dataset': args.dataset_name
        }

        if iteration >= (args.milestones[-1] + args.additional_step):
            save_checkpoint(rec_dict, is_best=False, filename='%d_step.tar' % iteration)
            print("Train finished!")
            break

        if best_EPE < 0:
            best_EPE = val_avg_sem
        is_best = val_avg_sem < best_EPE
        best_EPE = min(val_avg_sem, best_EPE)
        save_checkpoint(rec_dict, is_best)


def train(train_loader, model, optimizer, epoch, train_writer, init_spixl_map_idx, xy_feat):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loss_slic = AverageMeter()
    train_loss_sem = AverageMeter()
    train_loss_pos = AverageMeter()

    if args.epoch_size == 0:
        epoch_size = len(train_loader)
    else:
        epoch_size = min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()
    timestamp = time.time()
    iteration = 0

    for i, (image, label) in enumerate(train_loader):
        iteration = i + epoch * epoch_size
        image_gpu = image.to(device)
        label_gpu = label.to(device)

        # ========== adjust lr if necessary  ===============
        if (iteration + 1) in args.milestones:
            state_dict = optimizer.state_dict()
            for param_group in state_dict['param_groups']:
                param_group['lr'] = args.lr * (0.5 ** (args.milestones.index(iteration + 1) + 1))
            optimizer.load_state_dict(state_dict)

        # ========== complete data loading ================
        label_1hot = label2one_hot_torch(label_gpu, C=50)  # set C=50 as SSN does
        LABXY_feat_tensor = build_LABXY_feat(label_1hot, xy_feat)  # B* (50+2 )* H * W
        torch.cuda.synchronize()
        data_time.update(time.time() - timestamp)

        # ========== predict association map ============
        output = model(image_gpu)
        slic_loss, loss_sem, loss_pos = compute_semantic_pos_loss(output, LABXY_feat_tensor,
                                                                  pos_weight=args.pos_weight,
                                                                  kernel_size=args.downsize)

        # ========= back propagate ===============
        optimizer.zero_grad()
        slic_loss.backward()
        optimizer.step()

        # ========  measure batch time ===========
        torch.cuda.synchronize()
        batch_time.update(time.time() - timestamp)
        timestamp = time.time()

        # =========== record and display the loss ===========
        # record loss and EPE
        train_loss_slic.update(slic_loss.item(), image_gpu.size(0))
        train_loss_sem.update(loss_sem.item(), image_gpu.size(0))
        train_loss_pos.update(loss_pos.item(), image_gpu.size(0))

        if i % args.print_freq == 0:
            print('train Epoch: [{0}][Iteration{1}/Epoch_size{2}]\t batch_time {3}\t data_time {4}\t '
                  'train_loss_slic {5}\t train_loss_sem {6}\t train_loss_pos {7}\t'
                  .format(epoch, i, epoch_size, batch_time, data_time, train_loss_slic, train_loss_sem, train_loss_sem))

            train_writer.add_scalar('train_loss_slic', slic_loss.item(), i + epoch * epoch_size)
            train_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], i + epoch * epoch_size)

        n_iter += 1
        if i >= epoch_size:
            break
        if iteration >= (args.milestones[-1] + args.additional_step):
            break

    # =========== write information to tensorboard ===========
    if epoch % args.record_freq == 0:
        train_writer.add_scalar('train_loss_slic', slic_loss.item(), epoch)
        train_writer.add_scalar('train_loss_sem', loss_sem.item(), epoch)
        train_writer.add_scalar('train_loss_pos', loss_pos.item(), epoch)

        # save image
        mean_value = torch.tensor([0.411, 0.432, 0.45], dtype=image_gpu.dtype).view(3, 1, 1)
        image_l_save = (make_grid((image + mean_value).clamp(0, 1), nrow=args.batch_size))
        label_save = make_grid(args.label_factor * label)

        train_writer.add_image('Image', image_l_save, epoch)
        train_writer.add_image('Label', label_save, epoch)

        curr_spixl_map = update_spixl_map(init_spixl_map_idx, output)
        spixel_l_save = make_grid(curr_spixl_map, nrow=args.batch_size)[0, :, :]
        spixel_viz, _ = get_spixel_image(image_l_save, spixel_l_save)
        train_writer.add_image('Spixel viz', spixel_viz, epoch)

        # save associ map,  --- for debug only
        # _, prob_idx = torch.max(output, dim=1, keepdim=True)
        # prob_map_save = make_grid(assign2uint8(prob_idx))
        # train_writer.add_image('assigment idx', prob_map_save, epoch)

        print('==> write train step %dth to tensorboard' % i)

    return train_loss_slic.avg, train_loss_sem.avg, iteration


def validate(val_loader, model, epoch, val_writer, init_spixl_map_idx, xy_feat):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()

    val_loss_slic = AverageMeter()
    val_loss_sem = AverageMeter()
    val_loss_pos = AverageMeter()

    # set the validation epoch-size, we only randomly val. 400 batches during training to save time
    epoch_size = min(len(val_loader), 400)

    # switch to validation mode
    model.eval()
    timestamp = time.time()

    for i, (image, label) in enumerate(val_loader):
        image_gpu = image.to(device)
        label_gpu = label.to(device)

        # measure data loading time
        label_1hot = label2one_hot_torch(label_gpu, C=50)
        LABXY_feat_tensor = build_LABXY_feat(label_1hot, xy_feat)  # B* 50+2 * H * W
        torch.cuda.synchronize()
        data_time.update(time.time() - timestamp)

        # compute output
        with torch.no_grad():
            output = model(image_gpu)
            slic_loss, loss_sem, loss_pos = compute_semantic_pos_loss(output, LABXY_feat_tensor,
                                                                      pos_weight=args.pos_weight,
                                                                      kernel_size=args.downsize)

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - timestamp)
        timestamp = time.time()

        # record loss and EPE
        val_loss_slic.update(slic_loss.item(), image_gpu.size(0))
        val_loss_sem.update(loss_sem.item(), image_gpu.size(0))
        val_loss_pos.update(loss_pos.item(), image_gpu.size(0))

        if i % args.print_freq == 0:
            print('val Epoch: [{0}][Iteration {1}/ Epoch_size {2}]\t batch_time {3}\t data_time {4}\t '
                  'val_loss_slic {5}\t val_loss_sem {6}\t val_loss_pos {7}\t'
                  .format(epoch, i, epoch_size, batch_time, data_time, val_loss_slic, val_loss_sem, val_loss_pos))
        if i >= epoch_size:
            break

        # =============  write result to tensorboard ======================
        if epoch % args.record_freq == 0:
            val_writer.add_scalar('train_loss_epoch', slic_loss.item(), epoch)
            val_writer.add_scalar('loss_sem', loss_sem.item(), epoch)
            val_writer.add_scalar('loss_pos', loss_pos.item(), epoch)

            mean_value = torch.tensor([0.411, 0.432, 0.45], dtype=image_gpu.dtype).view(3, 1, 1)
            image_l_save = (make_grid((image + mean_value).clamp(0, 1), nrow=args.batch_size))

            curr_spixl_map = update_spixl_map(init_spixl_map_idx, output)
            spixel_l_save = make_grid(curr_spixl_map, nrow=args.batch_size)[0, :, :]
            spixel_viz, _ = get_spixel_image(image_l_save, spixel_l_save)

            label_save = make_grid(args.label_factor * label)
            val_writer.add_image('Image', image_l_save, epoch)
            val_writer.add_image('Label', label_save, epoch)
            val_writer.add_image('Spixel viz', spixel_viz, epoch)

            # --- for debug
            #     _, prob_idx = torch.max(assign, dim=1, keepdim=True)
            #     prob_map_save = make_grid(assign2uint8(prob_idx))
            #     val_writer.add_image('assigment idx level %d' % j, prob_map_save, epoch)

            print('==> write val step %dth to tensorboard' % i)

    return val_loss_slic.avg, val_loss_sem.avg


def save_checkpoint(state, is_best, filename='checkpoint.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.tar'))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
