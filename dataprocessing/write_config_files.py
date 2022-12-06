# Copyright (c) 2022, Elekta
# Björn Ahlgren

import os
import re
from natsort import natsorted
import argparse

def setup_arg_parser() :

    parser = argparse.ArgumentParser(description='Create config files for DeepMedic with relative paths to data files based on certain naming patterns and paths.')
    parser.add_argument('-data_path', dest='data_path', type=str,
                        help='Path to directory holding directory with nifti image files.')
    parser.add_argument('-output_path', dest='output_path', type=str, default='../config-generated',
                        help="Path to output directory to store config files. Creates directory of it does not exist")
    parser.add_argument('-postfix', dest='postfix', type=str, default='',
                        help="Postfix to append to all files created")
    args = parser.parse_args()
    return parser


def write_names(file_list, base_path, out_name):
    file_list = natsorted(file_list)
    with open(out_name, 'w') as fo:
        for f in file_list:
            line = f'{os.path.join(base_path, f)}\n'
            # line = f'../Data/{directory}/{f}\n'
            fo.write(line)

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Get the args
    data_path = args.data_path
    output_path = args.output_path
    postfix = args.postfix

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f'found data sub directories {os.listdir(data_path)}')
    for directory in os.listdir(data_path):
        base_path = f'../Data/{directory}/'

        if directory[-3:]=='val':
            file_list_val = os.listdir(os.path.join(data_path, directory))
            out_name = os.path.join(output_path, f'validationChannels{postfix}.cfg')
            
            write_names(file_list_val, base_path, out_name)

        if directory[-5:]=='train':
            file_list_train = os.listdir(os.path.join(data_path, directory))
            out_name = os.path.join(output_path, f'trainChannels{postfix}.cfg')
            write_names(file_list_train, base_path, out_name)

        if directory[-3:]=='seg':
            # get all seg files corresponding to validation files. use regex. then write to file. then repeat for training.
            file_list_seg = os.listdir(os.path.join(data_path, directory))
            dir_seg = directory

    # import pdb; pdb.set_trace()
    file_list_seg_val = []
    file_list_seg_train = []
    for f in file_list_seg:
        for fv in file_list_val:
            if re.match(f'{f[:-10]}.*',fv)!=None:
                file_list_seg_val.append(f)
        for ft in file_list_train:
            if re.match(f'{f[:-10]}.*',ft)!=None:
                file_list_seg_train.append(f)
        
    # directory = dir_seg
    

    file_list_preds = []
    for f in file_list_val:
        pred_name = f'pred_{f}'
        file_list_preds.append(f)
    out_name = os.path.join(output_path, f'validationNamesOfPredictions{postfix}.cfg')
    write_names(file_list_preds, base_path, out_name)

    base_path = f'../Data/{dir_seg}/'
    if len(file_list_seg_val) > 0:
        out_name = os.path.join(output_path, f'validationGtLabels{postfix}.cfg')
        write_names(file_list_seg_val, base_path, out_name)

    if len(file_list_seg_train) > 0:
        out_name = os.path.join(output_path, f'trainGtLabels{postfix}.cfg')
        write_names(file_list_seg_train, base_path, out_name)



if __name__ == "__main__":
    main()