import argparse
import os
import shutil
import time
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import models
import flow_transforms
from tensorboardX import SummaryWriter
from train_util import *
from imageio import imread, imsave
from skimage import img_as_ubyte

import sys
sys.path.append('./third_party/cython')
from connectivity import enforce_connectivity

'''
Infer from bsds500 dataset:
author:Fengting Yang 
last modification:  Mar.14th 2019

usage:
1. set the ckpt path (--pretrained_model) and output
2. comment the output if do not need

results will be saved at the args.output
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# ========================= training setting =========================
parser.add_argument('--dump_root', metavar='DIR', default='./dump_root',
                    help='path to images folder')
parser.add_argument('--pretrained_model', metavar='PTH', help='path to pre-trained model',
                    default='./pretrained_ckpt/SpixelNet_bsd_ckpt.tar')
parser.add_argument('--save_path', metavar='DIR', default='./test_result',
                    help='path to output folder')
parser.add_argument('--downsize', default=16, type=float,
                    help='superpixel grid cell, must be same as training setting')
parser.add_argument('-b', '--batch_size', default=8, type=int,  metavar='N',
                    help='mini-batch size')

# the BSDS500 has two types of image, horizontal and vertical one,
# here I use hor_img and ver_img to presents them respectively
parser.add_argument('--train_img_height', '-t_img_h', default=320, type=int, help='img height must be 16*n')
parser.add_argument('--train_img_width', '-t_img_w', default=480, type=int, help='img width must be 16*n')
parser.add_argument('--val_img_height', '-v_img_h', default=480, type=int, help='img height_must be 16*n')  #
parser.add_argument('--val_img_width', '-v_img_w', default=320, type=int, help='img width must be 16*n')

args = parser.parse_args()
args.test_list = args.dump_root + '/test.txt'
random.seed(100)


@torch.no_grad()
def test(model, test_list, save_path, spix_id, n, scale):
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = test_list[n]
    imgId = os.path.basename(img_file)[:-4]

    # origin size 481*321 or 321*481
    ori_img = imread(img_file)
    ori_h, ori_w, _ = ori_img.shape

    # choose the right spix_id
    if ori_h == 321 and ori_w == 481:
        spix_map_idx_tensor = spix_id[0]
        img = cv2.resize(ori_img, (int(480 * scale), int(320 * scale)), interpolation=cv2.INTER_CUBIC)
    elif ori_h == 481 and ori_w == 321:
        spix_map_idx_tensor = spix_id[1]
        img = cv2.resize(ori_img, (int(320 * scale), int(480 * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        print('The image size is wrong!')
        return

    img = input_transform(img)  # [3, 96, 144]
    ori_img = input_transform(ori_img)  # [3, 321, 481]
    # mean_value.shape: [3, 1, 1]
    mean_value = torch.tensor([0.411, 0.432, 0.45], dtype=img.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    # compute output
    tic = time.time()
    # output: [1, 9, 96, 144]
    output = model(img.cuda().unsqueeze(0))

    # assign the spixel map and  resize to the original size
    # curr_spix_map: [8, 1, 96, 144]
    # ori_sz_spix_map: [8, 1, 321, 481]
    curr_spix_map = update_spixl_map(spix_map_idx_tensor, output)
    ori_sz_spix_map = F.interpolate(curr_spix_map.type(torch.float), size=(ori_h, ori_w), mode='nearest').type(torch.int)

    # torch.Size([8, 321, 481]) # (8, 321, 481)
    spix_index_np = ori_sz_spix_map.squeeze().detach().cpu().numpy()
    spix_index_np = spix_index_np.astype(np.int64)
    segment_size = (spix_index_np.shape[1] * spix_index_np.shape[2]) / (int(600*scale*scale) * 1.0)
    min_size = int(0.5 * segment_size)
    max_size = int(10 * segment_size)
    spixel_label_map = enforce_connectivity(spix_index_np, min_size, max_size)[0]

    torch.cuda.synchronize()
    toc = time.time() - tic

    n_spixel = len(np.unique(spixel_label_map))
    given_img_np = (ori_img + mean_value).clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    spixel_bd_image = mark_boundaries(given_img_np / np.max(given_img_np), spixel_label_map.astype(int), color=(0, 1, 1))
    spixel_viz = spixel_bd_image.astype(np.float32).transpose(2, 0, 1)

    # ************************ Save all result********************************************
    # save img, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'img')):
    #     os.makedirs(os.path.join(save_path, 'img'))
    # spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
    # img_save = (ori_img + mean_values).clamp(0, 1)
    # imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))

    # save spixel viz
    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
    imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))

    # save the unique maps as csv for eval
    if not os.path.isdir(os.path.join(save_path, 'map_csv')):
        os.makedirs(os.path.join(save_path, 'map_csv'))
    map_csv_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
    # plus 1 to make it consistent with the toolkit format
    np.savetxt(map_csv_path, (spixel_label_map + 1).astype(int), fmt='%i', delimiter=",")

    if n % 10 == 0:
        print("processing %d" % n)

    return toc, n_spixel


def main():
    global args, save_path
    dump_root = args.dump_root
    print("=> fetching img pairs in '{}'".format(dump_root))

    train_img_height = args.train_img_height
    train_img_width = args.train_img_width
    val_img_height = args.val_img_height
    val_img_width = args.val_img_width

    mean_time_list = []
    # The spixel number we test

    for scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]:
        assert (320 * scale % 16 == 0 and 480 * scale % 16 == 0)
        save_path = args.save_path + '/bsds500/Spixel_{0}'.format(int(20 * scale * 30 * scale))

        args.train_img_height, args.train_img_width = train_img_height*scale, train_img_width*scale
        args.val_img_height, args.val_img_width = val_img_height*scale, val_img_width*scale

        print('=> save to result_path: {}'.format(save_path))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        test_list = []
        with open(args.test_list, 'r') as tf:
            img_path = tf.readlines()
            for path in img_path:
                test_list.append(path[:-1])

        print('The {} samples found'.format(len(test_list)))

        # create model
        network = torch.load(args.pretrained_model)
        print("=> use pretrained model '{}'".format(network['arch']))
        model = models.__dict__[network['arch']](data=network).cuda()
        model.eval()
        args.arch = network['arch']
        cudnn.benchmark = True

        # for vertical and horizontal input separately
        spix_id_1, _ = init_spixel_grid(args, b_train=True)
        spix_id_2, _ = init_spixel_grid(args, b_train=False)

        mean_time = 0
        # the following code is for debug
        for n in range(len(test_list)):
            time, n_spix = test(model, test_list, save_path, [spix_id_1, spix_id_2], n, scale)
            mean_time += time
        mean_time /= len(test_list)
        mean_time_list.append((n_spix, mean_time))

        print("for spixel number {}: with mean_time {} , generate {} spixels".format(int(20 * scale * 30 * scale), mean_time, n_spix))

    with open(args.save_path + '/SPixelNet/mean_time.txt', 'w+') as f:
        for item in mean_time_list:
            tmp = "n_spix:{}, mean_time:{}\n".format(item[0], item[1])
            f.write(tmp)


if __name__ == '__main__':
    main()
