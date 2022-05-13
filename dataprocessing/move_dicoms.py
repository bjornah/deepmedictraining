import os
import shutil

dicom_master_path = '/home/admin/Projects/Deployments/DeepMedicTraining/Data/dicom'

for dicom_folder in os.listdir(dicom_master_path):
    folder_path = os.path.join(dicom_master_path, dicom_folder)

    for f in os.listdir(folder_path):
        
        rtss_folder = os.path.join(dicom_master_path, f'{dicom_folder}_rtss')
        if not os.path.exists(rtss_folder):
            os.makedirs(rtss_folder)

        if f in ['RTDOSE.dcm', 'RTPLAN.dcm']:
            os.remove(os.path.join(folder_path, f))

        if f=='RTSS.dcm':
            shutil.move(os.path.join(folder_path, f), os.path.join(rtss_folder, f))