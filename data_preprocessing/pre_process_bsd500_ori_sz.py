import os
import numpy as np
import cv2
from scipy.io import loadmat
import argparse
from glob import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from pre_process_bsd500 import convert_label, BSD_loader
'''
Convert .mat data to the image and label data for test 

'''
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='/home/dulab/Documents/BSR_full', help="where the dataset is stored")
parser.add_argument("--dump_root", type=str, default="/home/dulab/Documents/super_pixel/my_sfcn/dump_root", help="Where to dump the data")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

'''
generate the ground truth TEST data :
label.png,  map.csv, and label_viz.png 
If we use ./superpixel-benchmark-master/examples/bash/my_run_eval_ours.sh for eval, we only need map.csv 

author: Fengting Yang 
last modified: Mar.15th 2019

Note, the image.jpg and gt .mat is copy from the ssn_superpixels-master
'''


def make_dataset(dir):
    cwd = os.getcwd()
    tst_list_path = os.path.join(cwd, 'test.txt')
    tst_list = []

    try:
        with open(tst_list_path, 'r') as tstf:
            tst_list_0 = tstf.readlines()
            for path in tst_list_0:
                img_path = os.path.join(dir, 'BSR/BSDS500/data/images/test', path[:-1]+'.jpg')
                if not os.path.isfile(img_path):
                    print('The validate images are missing in {}'.format(os.path.dirname(img_path)))
                    print('Please pre-process the BSDS500 as README states and provide the correct dataset path.')
                    exit(1)
                tst_list.append(img_path)

    except IOError:
        print('Error No avaliable list ')
        return
    return tst_list

def dump_example(n, n_total, data_type, img_path):
    global args
    if n % 100 == 0:
        print('Progress {0} {1}/{2}....'.format(data_type, n, n_total))

    img, label_lst = BSD_loader(img_path, img_path.replace('images', 'groundTruth')[:-4]+'.mat')
    dump_dir = os.path.join(args.dump_root_dir, data_type)
    if not os.path.isdir(dump_dir):
        try:
            os.makedirs(dump_dir)
        except OSError:
            if not os.path.isdir(dump_dir):
                raise

    img_name = os.path.basename(img_path)[:-4]
    for k, label in enumerate(label_lst):
        # save images
        dump_img_file = os.path.join(dump_dir, '{0}_img.jpg'.format(img_name))
        if not os.path.isfile(dump_img_file):
            cv2.imwrite(dump_img_file, img.astype(np.uint8))

        # save png label, uncomment it if needed
        # dump_label_file = os.path.join(dump_dir, '{0}_{1}_label.png' .format(img_name, k))
        # cv2.imwrite(dump_label_file, label.astype(np.uint8))

        # save label viz, uncomment if needed 
        # if not os.path.isdir(os.path.join(dump_dir,'label_viz')):
        #     os.makedirs(os.path.join(dump_dir,'label_viz'))
        # dump_label_viz = os.path.join(dump_dir, 'label_viz',  '{0}_{1}_label_viz.png'.format(img_name, k))
        # plt.imshow(label) #val2uint8(tgt_disp, MAX_DISP, MIN_DISP)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig(dump_label_viz,bbox_inches='tight',pad_inches=0)
        # plt.close()

        # save csv for formal evaluation with benchmark code
        if not os.path.isdir(os.path.join(dump_dir, 'map_csv')):
            os.makedirs(os.path.join(dump_dir, 'map_csv'))
        dump_label_csv = os.path.join(dump_dir, 'map_csv', '{0}-{1}.csv'.format(img_name, k))
        np.savetxt(dump_label_csv, (label + 1).astype(int), fmt='%i', delimiter=",")


def main():
    data_dir = args.dataset_dir
    test_list = make_dataset(data_dir)
    
    dump_dir= os.path.abspath(args.dump_root)
    print("data will be saved to {}".format(dump_dir))
    # single thread for debug
    # for n, train_samp in enumerate(train_list):
    #     dump_example(n, len(train_list),'train', train_samp)

    # multi-thread running for speed
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, len(test_list), 'test', test_samp) for n, test_samp in enumerate(test_list))

    with open(dump_dir + '/test.txt', 'w') as trnf:
        imfiles = glob(os.path.join(dump_dir, 'test', '*_img.jpg'))
        for frame in imfiles:
            trnf.write(frame + '\n')


if __name__ == '__main__':
    main()
