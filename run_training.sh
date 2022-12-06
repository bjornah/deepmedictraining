#!/bin/bash

# create output folder based on date and time
T=$(date +"%Y-%m-%d-%H-%M")
mkdir /home/admin/Projects/Deployments/DeepMedicTraining/output/$T

    # -v /home/admin/Projects/Deployments/DeepMedicTraining/Data:/home/Data \

docker run --rm --gpus all \
    -v /home/admin/Data/Brats:/home/Data \
    -v /home/admin/Projects/Deployments/DeepMedicTraining/output/$T:/home/output \
    -v /home/admin/Projects/Deployments/DeepMedicTraining/config-docker:/home/config \
    deepmedictraining

