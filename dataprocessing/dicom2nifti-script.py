import argparse
import numpy as np
import SimpleITK as sitk
import os
import dicom2nifti
import nibabel as nib
from rt_utils import RTStructBuilder
from typing import Tuple
import numpy as np
import logging
from glob import glob
import pandas as pd
import yaml
from dicom_nifti_conversion import process_dicom_rtss_to_nifti
# from preprocessing import itk_preprocessing


def setup_arg_parser() :

    parser = argparse.ArgumentParser(description='Automatic conversion from dicom to nifti files. Please inspect the results to make sure that they are correct.')
    parser.add_argument('-input_path', dest='input_path', type=str,
                        help='Path to directory holding directories with dicom images. \
                            Files should be organised as input_path/pat*/*.dcm and \
                            input_path/pat*_rtss/RTSS.dcm')
    parser.add_argument('-output', dest='output_path', type=str, default='output',
                        help="Path to output directory to store nifti files")
    parser.add_argument('-structure_map', dest='structure_map', type=str, default='dataprocessing/structure_map.yml',
                        help="Path to yaml file storing mapping for different roi names to label classes")

    args = parser.parse_args()
    return parser

def main():
    logging.basicConfig(
        filename=f'dicom-conversions.log', 
        format='%(levelname)s:%(message)s',
        level=logging.DEBUG,
        filemode='w')

    logging.info(f'Started')

    parser = setup_arg_parser()
    args = parser.parse_args()

    logging.debug(f'parsed args: {args}')
    # Get the args
    input_path = args.input_path
    output_path = args.output_path
    structure_map = args.structure_map

    # structure_map = '/home/admin/Projects/Deployments/DeepMedicTraining/dataprocessing/structure_map.yml'
    with open(structure_map) as fh:
        structure_map = yaml.load(fh, Loader=yaml.FullLoader)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logging.info(f'Created {output_path}')

##################################
    dicom_dir_list = []
    dicom_rtss_list = []
    for dicom_dir in os.listdir(input_path):
        
        if dicom_dir.split('_')[-1] != 'rtss':
            dicom_dir_list.append(dicom_dir)

        if dicom_dir.split('_')[-1] == 'rtss':
            dicom_rtss_list.append(os.path.join(input_path, dicom_dir, 'RTSS.dcm'))

        # for f in os.listdir(folder_path):
            
        #     rtss_folder = os.path.join(dicom_master_path, f'{dicom_folder}_rtss')
        #     if not os.path.exists(rtss_folder):
        #         os.makedirs(rtss_folder)

        #     if f in ['RTDOSE.dcm', 'RTPLAN.dcm']:
        #         os.remove(os.path.join(folder_path, f))

        #     if f=='RTSS.dcm':
        #         shutil.move(os.path.join(folder_path, f), os.path.join(rtss_folder, f))

##################################

    number_of_patients = len(dicom_dir_list)
    logging.info(f'Found {number_of_patients} dicom folders in {input_path}')
    logging.info(f'Converting to nifti files')
    dicom_spacing_dir = {}
    for i,dicom_dir in enumerate(dicom_dir_list): #iterate through dirs in dicoms_path
        logging.debug(f'dicom folder: {dicom_dir}')
        nii_file_dir = dicom2nifti.dicom_series_to_nifti(
                        original_dicom_directory=os.path.join(input_path, dicom_dir), 
                        output_file=os.path.join(output_path, dicom_dir+'_nii'),
                        reorient_nifti=False
                        )

        dicom_spacing = nii_file_dir['NII'].header.get_zooms() # gets voxel spacing
        dicom_spacing = tuple([float(ds) for ds in dicom_spacing]) # this is needed to convert float32 to float. Otherwise sitk. Resample throws really confusing errors
        dicom_spacing_dir[dicom_dir] = dicom_spacing # save spacing so that output rtss can be converted back into this spacing
        logging.debug(f'{dicom_dir} has spacing {dicom_spacing}')
        logging.debug(f'{dicom_dir} converted to {os.path.join(output_path, dicom_dir+"_nii")}')
        logging.debug(f'{i+1}/{number_of_patients} patients dicom images processed')
    
    df = pd.DataFrame(dicom_spacing_dir)
    savename = os.path.join(output_path,'dicom_spacings.csv')
    df.to_csv(savename, index=False)
    logging.info(f'Saved dicom spacings to {savename}. This can be used to convert back to dicom files from nifti.')
    logging.info('Done with dicom image to nifti conversion')

    for i,rtss in enumerate(dicom_rtss_list):
        dicom_rtss_dir = rtss.split('/')[-2] # this is the dicom_dir
        dicom_name = dicom_rtss_dir[:-5]
        process_dicom_rtss_to_nifti(
            dicom_folder=os.path.join(input_path, dicom_name),
            dicom_rtss_path=rtss,
            output_nii_path=os.path.join(output_path, dicom_name+'_rtss_nii'),
            structure_map=structure_map
        )
        logging.debug(f'{rtss} converted to {os.path.join(output_path, dicom_name+"_rtss_nii")}')
        logging.debug(f'{i+1}/{number_of_patients} rtss files processed')

    logging.info('Done with dicom rtss to nifti conversion')

    # else:
    #     nifti_in = input_path
    #     nifti_out = output_path

    # logging.debug(f'os.getcwd() = {os.getcwd()}')
    # logging.debug(f'nifti_in = {nifti_in}')
    # logging.debug(f'os.listdir(nifti_in) = {os.listdir(nifti_in)}')

    # number_of_nifti_files = len(os.listdir(input_path))
    # logging.info(f'Performing prediction on {number_of_nifti_files} nifti files')
    # for i,img in enumerate(os.listdir(nifti_in)):
    #     logging.debug('---inference time---')
    #     logging.debug(f'nifti in file = {img}')
    #     try:
    #         dicom_name = img.split('.')[0][:-4]
    #         logging.debug(f'patient dicom name = {dicom_name}')

    #         nifti_rtss = os.path.join(nifti_out, f'pred-{img}.gz')
    #         process_img(os.path.join(nifti_in,img), nifti_rtss, pb_dir)
    #         logging.info('Prediction successful')
    #     except Exception as e: 
    #         logging.warning(f'Could not perform prediction on nifti file {img}, error msg = {e}')

    #     try:
    #         if data_are_dicom_dirs==True:
    #             dicom_spacing = dicom_spacing_dir[dicom_name]
    #             logging.debug(f'dicom spacing is {dicom_spacing}, for {dicom_name}')
    #             output_file = os.path.join(output_path,f'pred-{dicom_name}')
    #             original_dicom_folder = os.path.join(input_path, dicom_name)
    #             logging.debug('runnint itk postprocessing in the middle of main')
    #             logging.debug(f'nifti_rtss = {nifti_rtss}')
    #             rtss = sitk.ReadImage(nifti_rtss)
    #             rtss = itk_postprocessing(rtss, dicom_spacing)

    #             nifti_rtss_to_dicom_rtss(
    #                 nifti_rtss=nifti_rtss, 
    #                 original_dicom_folder=original_dicom_folder, 
    #                 output_dicom_rtss=output_file,
    #                 inference_threshold = inference_threshold,
    #                 new_spacing=dicom_spacing
    #                 )
    #             logging.info('Conversion from nifti to dicom, {output_file}, successful')
    #             logging.debug(f'converted back to {dicom_spacing} spacing')
    #     except Exception as e: 
    #         print(e)
    #         logging.warning(f'Could not convert nifti to dicom, error msg = {e}')

    #     logging.info(f'{i+1}/{number_of_nifti_files} done')

    # logging.info('Inference finished.')

if __name__ == "__main__":
    main()