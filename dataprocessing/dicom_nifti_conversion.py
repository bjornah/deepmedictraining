import logging
import SimpleITK as sitk
from typing import Tuple
import numpy as np
# from preprocessing import itk_preprocessing, itk_postprocessing
from rt_utils import RTStruct, RTStructBuilder
import nibabel as nib
# import dicom2nifti

# def load_nifti_img(nifti_file: str, preprocess: bool, new_spacing: tuple = (1.0, 1.0, 1.0)):
#     '''Loads nifti image using sitk and rotates it so that it matches dicom orientation.

#     Args:
#         nifti_file (str): Path to nifti file
#         preprocess (bool): Whether to apply preprocessing to image or not (z score normalisation) and casting to float32
#         new_spacing (tuple): The desired spatial spacing of the voxels. DeepMedic has been trained using a resolution of (1,1,1)
#     '''
#     nifti_image = sitk.ReadImage(nifti_file)
#     if preprocess==True:
#         nifti_image = itk_preprocessing(nifti_image, new_spacing=new_spacing)
#     else:
#         nifti_image = itk_postprocessing(nifti_image, new_spacing=new_spacing) # this just resamples the image
#     nifti_image_data = axes_swapping(sitk.GetArrayFromImage(nifti_image))
#     return nifti_image_data

# def load_nifti_pred(nifti_pred_file: str, new_spacing: tuple = (1.0, 1.0, 1.0)):
#     '''Loads nifti file with predictions using sitk and rotates it so that it matches dicom orientation.
    
#     Args:
#         nifti_file (str): Path to nifti file
#         new_spacing (tuple): The desired spatial spacing of the voxels. DeepMedic has been trained using a resolution of (1,1,1)'''
#     nifti_pred = sitk.ReadImage(nifti_pred_file)
#     nifti_pred = itk_postprocessing(nifti_pred, new_spacing=new_spacing) # this just resamples the image
#     nifti_pred_data = axes_swapping(sitk.GetArrayFromImage(nifti_pred))
#     return nifti_pred_data

def find_clusters_itk(mask: sitk.Image, max_object: int = 50) -> Tuple[sitk.Image, int]:
    """
    Find how many clusters in the mask (ITK Image).
    Args:
        mask: a binary sitk.Image
        max_object: a threshold to make sure the # of object does not exceed max_object.

    Return:
        label_image: a masked sitk.Image with cluster index
        num: number of clusters
    """

    # Make sure mask is a binary image
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Make sure they are binary
    min_max_f = sitk.MinimumMaximumImageFilter()
    min_max_f.Execute(mask)
    mask_max = min_max_f.GetMaximum()
    mask_min = min_max_f.GetMinimum()

    assert mask_max <= 1, f"mask should be binary, but the maximum value is {mask_max}"
    assert mask_min >= 0, f"mask should be binary, but the minimum value is {mask_min}"

    label_map_f = sitk.BinaryImageToLabelMapFilter()
    label_map = label_map_f.Execute(mask)

    label_map_to_image_f = sitk.LabelMapToLabelImageFilter()
    label_image = label_map_to_image_f.Execute(label_map)

    num = label_map_f.GetNumberOfObjects()

    assert (
        num <= max_object
    ), f"Does not expect there are clusters more than {max_object} but is {num}"

    return label_image, num

def axes_swapping(array: np.array):
    array = array.T
    array = np.swapaxes(array, 0, 1)
    # array = array[:,::-1,:]
    return array

def get_binary_rtss(nifti_pred: str, inference_threshold: float, new_spacing: tuple):
    # rtss = os.path.abspath(nifti_pred)
    logging.debug(f'rtss = {nifti_pred}')
    rtss = sitk.ReadImage(nifti_pred)

    rtss = itk_postprocessing(rtss, new_spacing) # resamples data to original dicom resolution before executing binary threshold
    
    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(inference_threshold)
    binary_filter.SetUpperThreshold(1)

    rtss_binary = binary_filter.Execute(rtss)

    return rtss_binary
    
def nifti_rtss_to_dicom_rtss(
    nifti_rtss,
    original_dicom_folder, 
    output_dicom_rtss, 
    inference_threshold: float, 
    new_spacing: tuple
    ):
    logging.debug(f'nifti_rtss = {nifti_rtss}')
    rtss_binary = get_binary_rtss(nifti_rtss, inference_threshold, new_spacing)

    label_image, n_target = find_clusters_itk(rtss_binary, max_object=200)
    logging.info(f'number of disjoint clusters/tumours in prediction: {n_target}')
    nda = sitk.GetArrayFromImage(label_image)
    nda = axes_swapping(nda)
    # nda = nda[::-1,:,:]
    
    logging.debug(f'unique labels in image: {np.unique(nda)}')

    rtstruct = RTStructBuilder.create_new(
                    dicom_series_path=original_dicom_folder,
                    )

    small_clusters = 0
    for idx in range(1,n_target+1): # 0 is background, so don't use that label as ROI
        ROI = np.where(nda==idx, 1, 0)
        mask = np.ma.make_mask(ROI)
        if mask.sum()<10:
            small_clusters+1
        rtstruct.add_roi(
            mask=mask, 
            color=[100, 150, int(255/n_target*idx)], 
            name=f"MET nr {idx}"
            )
    logging.debug(f'There are currently {small_clusters} clusters with fewer than 10 pixels.')

    rtstruct.save(output_dicom_rtss)

def copy_nifti_header(src: nib.Nifti1Image, dst: nib.Nifti1Image) -> nib.Nifti1Image:
    """Copy header from src to dst while perserving the dst data."""
    data = dst.get_fdata()
    return nib.nifti1.Nifti1Image(data, None, header=src.header)

def load_nifti_rtss(nifti_pred):
    rtss = sitk.ReadImage(nifti_pred)
    
    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(0.5)
    binary_filter.SetUpperThreshold(1)
    rtss_binary = binary_filter.Execute(rtss)

    nifti_rtss = sitk.GetArrayFromImage(rtss_binary)

    return nifti_rtss

def fetch_mapped_rois(rtstruct: RTStruct, structure_map: dict) -> np.ndarray:

    masks = {}
    roi_names = rtstruct.get_roi_names()

    for roi_idx, roi_name in enumerate(roi_names):

        for structure_idx, structures in structure_map.items():

            mask_idx = int(structure_idx)

            if roi_name.lower() not in (_.lower() for _ in structures):
                continue

            logging.debug(f"\t-- Converting structure: {roi_name}")

            try:
                mask = rtstruct.get_roi_mask_by_name(roi_name)

                number_voxel = np.count_nonzero(mask)
                logging.debug(f"\tCounted number of voxel: {number_voxel}")

                if number_voxel == 0:
                    continue

                if mask_idx in masks:
                    mask = np.logical_or(mask, masks[mask_idx])

                masks[mask_idx] = mask

                break

            except Exception as e:
                logging.error(e)
                break

    if len(masks) == 0:
        return None

    shape = masks[list(masks.keys())[0]].shape
    shape = shape + (len(structure_map) + 1,)
    stacked_mask = np.zeros(shape, dtype=np.uint8)

    for idx, mask in masks.items():
        stacked_mask[:, :, :, idx] = mask.astype(np.uint8)

    # Set background
    background = np.sum(stacked_mask, axis=-1) == 0
    stacked_mask[..., 0] = background

    return stacked_mask

def fetch_rtstruct_roi_masks(
    rtstruct: RTStruct,
    structure_map: dict = None,
) -> np.ndarray:
    """
    Default structure list start from 1
    """

    if structure_map is not None:
        masks = fetch_mapped_rois(rtstruct, structure_map)

    else:
        masks = fetch_all_rois(rtstruct)

    return masks

def fetch_all_rois(rtstruct: RTStruct) -> np.ndarray:

    masks = []
    roi_names = rtstruct.get_roi_names()

    for roi_name in roi_names:
        mask = rtstruct.get_roi_mask_by_name(roi_name)
        masks.append(mask)

    if len(masks) == 0:
        return None

    flat_mask = np.sum(masks, axis=0) > 0

    return flat_mask

def process_dicom_rtss_to_nifti(
    dicom_folder: str,
    dicom_rtss_path: str,
    output_nii_path: str,
    structure_map: dict = None,
) -> str:
    """
    """

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_folder,
        rt_struct_path=dicom_rtss_path,
    )

    rtss_mask = fetch_rtstruct_roi_masks(rtstruct, structure_map)
    rtss_mask = rtss_mask.astype(np.int8)

    # To match with the dicom2nifti.dicom_series_to_nifti orientation
    rtss_mask = np.swapaxes(rtss_mask, 0, 1)

    # Undo one-hot encoding. This makes the sitk resampling simpler, and is consistent with the data provided with the deepmedic repo.
    rtss_mask = np.argmax(rtss_mask, axis=-1)

    rtss_nii = nib.Nifti1Image(rtss_mask, affine=np.eye(4))
    nib.save(rtss_nii, output_nii_path)
    # except Exception as e:
    #     logging.error("Unable to convert dicom series to nifti: ", e)

    return output_nii_path



#################################################

# def process_dicom_rtss_to_nifti(
#     df_rtss: pd.DataFrame,
#     structure_map: Dict[int, Union[List, str]],
#     output_dir: str = None,
# ) -> str:
#     """
#     Process matched RT structure set dataframe that is grouped by keys
#     SeriesInstanceUID and RTSS_file.
#     """

#     assert bool("output_dir" in df_rtss.columns) != bool(
#         output_dir is not None
#     ), "Either provide output_dir as string or require output_dir in df_rtss."

#     if output_dir is None:
#         output_dir = list(df_rtss["output_dir"].unique())
#         assert (
#             len(output_dir) == 1
#         ), f"There should be 1 output_dir but is {len(output_dir)}."
#         output_dir = output_dir[0]
#     else:
#         assert isinstance(
#             output_dir, str
#         ), f"output_dir should be str but is {type(output_dir)}"

#     nii_file = None

    # with tempfile.TemporaryDirectory() as tmp_dir:

    #     df_rtss = df_rtss.reset_index()

    #     df_rtss["tmp_file"] = df_rtss.apply(
    #         lambda x: copy_file_safely(
    #             tmp_dir=tmp_dir,
    #             src=x["file"],
    #             dst_naming=x["SOPInstanceUID"],
    #         ),
    #         axis=1,
    #     )

    #     PatientID = list(df_rtss["PatientID"].unique())
    #     RTSS_file = list(df_rtss["RTSS_file"].unique())
    #     SeriesInstanceUID = list(df_rtss["SeriesInstanceUID"].unique())
    #     RTSS_SOPInstanceUID = list(df_rtss["RTSS_SOPInstanceUID"].unique())

    #     # To check all the above variables have only one item
    #     assert (
    #         len(PatientID) == 1
    #     ), f"There should be 1 PatientID but is {len(PatientID)}."
    #     assert (
    #         len(RTSS_file) == 1
    #     ), f"There should be 1 RTSS_file but is {len(RTSS_file)}."
    #     assert (
    #         len(SeriesInstanceUID) == 1
    #     ), f"There should be 1 SeriesInstanceUID but is {len(SeriesInstanceUID)}."
    #     assert (
    #         len(RTSS_SOPInstanceUID) == 1
    #     ), f"There should be 1 RTSS_SOPInstanceUID but is {len(RTSS_SOPInstanceUID)}."

    #     PatientID = PatientID[0]
    #     RTSS_file = RTSS_file[0]
    #     series_UID = RTSS_SOPInstanceUID[0]

    #     RTSS_tmp_file = copy_file_safely(
    #         tmp_dir=tmp_dir,
    #         src=RTSS_file,
    #         dst_naming=series_UID,
    #     )

    #     if RTSS_tmp_file is not None:
    #         try:

    #             logging.info(f"Converting RTSS for {PatientID} ({series_UID})...")

    #             output_fname = join_path(
    #                 [output_dir, PatientID, series_UID.replace(".", "")]
    #             )
    #             output_fname = output_fname + ".nii"
    #             make_sure_file_directory_exists(output_fname)

    #             rtstruct = RTStructBuilder.create_from(
    #                 dicom_series_path=tmp_dir,
    #                 rt_struct_path=RTSS_tmp_file,
    #             )

    #             rtss_mask = fetch_rtstruct_roi_masks(rtstruct, structure_map)
    #             rtss_mask = rtss_mask.astype(np.int8)

    #             # To match with the dicom2nifti.dicom_series_to_nifti orientation
    #             rtss_mask = np.swapaxes(rtss_mask, 0, 1)

    #             rtss_nii = nib.Nifti1Image(rtss_mask, affine=np.eye(4))
    #             nib.save(rtss_nii, output_fname)

    #             nii_file = output_fname

    #         except Exception as e:
    #             logging.error("Unable to convert dicom series to nifti: ", e)

    # return nii_file

