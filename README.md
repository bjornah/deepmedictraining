# README
Repo for creating docker image that trains DeepMedic model.

To convert dicom images and rtss to nifti files:

    python dicom2nifti-script.py -input_path .../Data/dicom -output .../Data/nifti-converted -structure_map .../dataprocessing/structure_map.yml

    .../ should be replaced by whatever full path is appropriate

When running training, first edit then execute run_training.sh. You need to edit the mount points to the docker container, i.e. the folders that will be available to the docker once it's running. These are the folders in which you provide the training and validation data (mounted to /home/Data), the config files reqired by deepmedic (mounted to /home/config), and the folder in which deepmedic saves output (mounted to /home/output). This boild down to needing to alter the following lines in the script:

    -v your_data_path_goes_here:/home/Data \
    -v your_output_path_goes_here/$T:/home/output \
    -v your_config_path_goes_here:/home/config \

If you make changes to the deepmedic repo and want to incorporate these into the docker image you run 
    
    > ./build.sh 
    
which creates a new image. Don't forget that you might want to remove old versions of the image using something like 
    
    > docker images prune. 

If you want to move the docker image to another machine you can run

    > docker save -o DeeeMedicDockerImage.tar deepmedictraining

Use something like scp to move it wherever you need it. On the new host machine, run

    > docker load -i DeeeMedicDockerImage.tar

and you should be good to go. If this does not work you can try copying the entire repo over and building it on the new machine. 