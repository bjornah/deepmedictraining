#!/bin/bash

# activate conda environment
# conda init bash
echo "checking cuda version"
nvcc --version
echo ""

echo "activate conda env"
source /root/anaconda/etc/profile.d/conda.sh
conda activate myenv

echo "Run training"
./home/deepmedic/deepMedicRun -model /home/config/modelConfig.cfg \
                         -train /home/config/trainConfig.cfg \
                         -dev cuda0


# possible alternative syntax
# conda run -n myenv python /home/deepmedic/deepMedicRun ...
