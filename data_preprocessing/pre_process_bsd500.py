import os
import numpy as np
import cv2
from scipy.io import loadmat
import argparse
from glob import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/dulab/Project/BSR_full",
                        help="where the filtered dataset is stored")
    parser.add_argument("--dump_root", type=str, default="/home/dulab/Project/sfcn_pytorch/dump_root",
                        help="Where to dump the data")
    parser.add_argument("--b_filter", type=bool, default=False,
                        help="we do not use this in our paper")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="number of threads to use")

    args = parser.parse_args()
    return args

'''
pre_process_bsd500.py: extract each pair of image and label from .mat to generate the TRAINING and VALIDATION data 
pre_process_bsd500_ori_sz.py: generate TEST data in the same folder 

We follow the SSN configuration to discard all samples that have more than 50 classes in their segments, and 
we use the exactly same train, val, and test list as SSN, see the train/val/test.txt in the data_preprocessing folder for details
  
author: Fengting Yang 
March. 1st 2019 

Modified: five
'''


def make_dataset_list(dir):
    cwd = os.getcwd()
    train_list_dir = cwd + '/train.txt'
    val_list_dir = cwd + '/val.txt'
    test_list_dir = cwd + '/test.txt'
    test_list = []
    train_list = []
    val_list = []

    try:
        with open(train_list_dir, 'r') as tf:
            train_img_id_list = tf.readlines()
            for img_id in train_img_id_list:
                img_dir = os.path.join(dir, 'BSR/BSDS500/data/images/train', img_id[:-1]+'.jpg')
                print(img_dir)
                if not os.path.isfile(img_dir):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_dir)))
                    print('Please pre-process the BSDS500 as README states and provide the correct dataset path.')
                    exit(1)
                train_list.append(img_dir)

        with open(val_list_dir, 'r') as vf:
            val_img_id_list = vf.readlines()
            for img_id in val_img_id_list:
                img_dir = os.path.join(dir, 'BSR/BSDS500/data/images/val', img_id[:-1] + '.jpg')
                if not os.path.isfile(img_dir):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_dir)))
                    print('Please pre-process the BSDS500 as README states and provide the correct dataset path.')
                    exit(1)
                val_list.append(img_dir)

        with open(test_list_dir, 'r') as tf:
            test_img_id_list = tf.readlines()
            for img_id in test_img_id_list:
                img_dir = os.path.join(dir, 'BSR/BSDS500/data/images/test', img_id[:-1]+'.jpg')
                if not os.path.isfile(img_dir):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_dir)))
                    print('Please pre-process the BSDS500 as README states and provide the correct dataset path.')
                    exit(1)
                test_list.append(img_dir)
    except IOError:
        print('Error. No available list.')
        return

    return train_list, val_list, test_list


def convert_label(label):
    prob_label = np.zeros((label.shape[0], label.shape[1], 50)).astype(np.float32)
    ct = 0
    total_seg_num = [np.unique(label)]
    for seg_index in total_seg_num:
        if ct >= 50:
            print('give up sample because label shape is larger than 50: {0}'.format(len(total_seg_num)))
            break
        else:
            prob_label[:, :, ct] = (label == seg_index)  # one hot
        ct = ct + 1

    # print(prob_label[:, :, 0].shape)
    # print(prob_label[:, :, 0])
    label2 = np.squeeze(np.argmax(prob_label, axis=-1))

    # np.savetxt('data.csv', label2, delimiter=',', fmt='%d')
    return label2, prob_label


def BSD_loader(img_dir, label_dir, b_filter=False):
    '''
    Process one image and its corresonding label.
    :param img_dir: The image name.
    :param label_dir: The label name.
    :param b_filter: Whether or not use bilateral filter.
    :return:
    '''
    img_ = cv2.imread(img_dir)

    # origin size 481*321 or 321*481
    H_, W_, _ = img_.shape

    # crop to 16*n size
    if H_ == 321 and W_ == 481:
        img = img_[:320, :480, :]
    elif H_ == 481 and W_ == 321:
        img = img_[:480, :320, :]
    else:
        print('It is not BSDS500 images.')
        exit(1)

    if b_filter:
        img = cv2.bilateralFilter(img, 5, 75, 75)

    gt_seg_list = []
    # dict_keys(['__header__', '__version__', '__globals__', 'groundTruth'])
    # shape(1, 6)
    gt_seg_dict = loadmat(label_dir)
    # gt_seg_dict['groundTruth'][0][1][0][0][0] # Segmentation (321, 481)
    # gt_seg_dict['groundTruth'][0][1][0][0][1] # Boundaries (321, 481)
    for label_num in range(len(gt_seg_dict['groundTruth'][0])):
        # Get the ground truth segmentation part
        gt_seg = gt_seg_dict['groundTruth'][0][label_num][0][0][0]
        label_, prob_label = convert_label(gt_seg)
        if H_ == 321 and W_ == 481:
            label = label_[:320, :480]
        elif H_ == 481 and W_ == 321:
            label = label_[:480, :320]
        gt_seg_list.append(label)

    return img, gt_seg_list


def dump_example(args, index, num, img_dir, dataset_type):
    if index % 100 == 0:
        print('Progress {0} {1}/{2}....' .format(dataset_type, index, num))

    img, gt_seg_list = BSD_loader(img_dir, img_dir.replace('images', 'groundTruth')[:-4]+'.mat', b_filter=args.b_filter)

    if args.b_filter:
        dump_dir = os.path.join(args.dump_root, dataset_type + '_b_filter_' + str(args.b_filter))
    else:
        dump_dir = os.path.join(args.dump_root, dataset_type)
    if not os.path.isdir(dump_dir):
        try:
            os.makedirs(dump_dir)
        except OSError:
            if not os.path.isdir(dump_dir):
                raise

    img_name = os.path.basename(img_dir)[:-4]
    if dataset_type == 'train' or 'val':
        for label_index, label in enumerate(gt_seg_list):
            # Save the same image five times
            dump_img_file = os.path.join(dump_dir,  '{0}_{1}_img.jpg' .format(img_name, label_index))
            cv2.imwrite(dump_img_file, img.astype(np.uint8))

            # save labels with different label indexs
            dump_label_file = os.path.join(dump_dir, '{0}_{1}_label.png' .format(img_name, label_index))
            cv2.imwrite(dump_label_file, label.astype(np.uint8))

    else:
        print(dataset_type)
        dump_img_file = os.path.join(dump_dir, '{0}_img.jpg'.format(img_name))
        if not os.path.isfile(dump_img_file):
            cv2.imwrite(dump_img_file, img.astype(np.uint8))
        if not os.path.isdir(os.path.join(dump_dir, 'map_csv')):
            os.makedirs(os.path.join(dump_dir, 'map_csv'))
        for label_index, label in enumerate(gt_seg_list):
            # save csv for formal evaluation with benchmark code
            dump_label_csv = os.path.join(dump_dir, 'map_csv', '{0}_{1}.csv'.format(img_name, label_index))
            np.savetxt(dump_label_csv, (label + 1).astype(int), fmt='%i', delimiter=",")

        # save label viz, uncomment if needed
        # if not os.path.isdir(os.path.join(dump_dir,'label_viz')):
        #     os.makedirs(os.path.join(dump_dir,'label_viz'))
        # dump_label_viz = os.path.join(dump_dir, 'label_viz',  '{0}_{1}_label_viz.png'.format(img_name, k))
        # plt.imshow(label)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig(dump_label_viz,bbox_inches='tight',pad_inches=0)
        # plt.close()


def main(args):
    data_dir = args.dataset_dir
    train_list, val_list, test_list = make_dataset_list(data_dir)
    print(test_list)

    dump_dir = os.path.abspath(args.dump_root)
    print("Preprocessed data will be saved to {}".format(dump_dir))
    # for debug only
    # for index, train_img_dir in enumerate(train_list):
    #     dump_example(args, index, len(train_list), train_img_dir, dataset_type='train')
    # for index, val_img_dir in enumerate(val_list):
    #     dump_example(args, index, len(val_list), val_img_dir, dataset_type='val')
    # for index, test_img_dir in enumerate(test_list):
    #     dump_example(args, index, len(test_list), test_img_dir, dataset_type='test')

    # multi-thread running for speed
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(args, index, len(train_list), train_samp, 'train') for index, train_samp in enumerate(train_list))
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(args, n, len(val_list), val_samp, 'val') for n, val_samp in enumerate(val_list))
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(args, n, len(test_list), test_samp, 'test') for n, test_samp in enumerate(test_list))

    with open(dump_dir + '/train.txt', 'w') as trnf:
        imfiles = glob(os.path.join(dump_dir, 'train', '*_img.jpg'))
        for frame in imfiles:
            trnf.write(frame + '\n')

    with open(dump_dir + '/val.txt', 'w') as trnf:
        imfiles = glob(os.path.join(dump_dir, 'val', '*_img.jpg'))
        for frame in imfiles:
            trnf.write(frame + '\n')

    with open(dump_dir + '/test.txt', 'w') as trnf:
        imfiles = glob(os.path.join(dump_dir, 'test', '*_img.jpg'))
        for frame in imfiles:
            trnf.write(frame + '\n')



if __name__ == '__main__':
    args = parse_arguments()
    main(args)