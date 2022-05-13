# Start from CentOS 7
# FROM nvidia/cuda:10.0-base
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Set the shell to bash
SHELL ["/bin/bash", "-c"]
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub # aparently there is some issue with keys from nvidia at the moment and this fixes it #2022-05-10
# Install needed packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y unzip
RUN apt-get install ffmpeg libsm6 libxext6 -y
# RUN apt-get install nvidia-cuda-toolkit -y

# Install Anaconda
RUN cd /home && \
  wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh && \
  chmod +x Anaconda3-2019.07-Linux-x86_64.sh && \
  ./Anaconda3-2019.07-Linux-x86_64.sh -b -p ~/anaconda && \
  rm Anaconda3-2019.07-Linux-x86_64.sh

# Add Anaconda to path
ENV PATH=/root/anaconda/bin:$PATH

# Create keras conda environment
COPY requirement.txt /home
RUN conda create --name myenv python=3.6.8

# Add conda environment to path
ENV PATH=/root/anaconda/envs/myenv/bin:$PATH

# Copy deepmedic
COPY deepmedic /home/deepmedic

# Create mount point folders
RUN mkdir /home/Data
RUN mkdir /home/output
RUN mkdir /home/config

# copy pipeline script and make executable
COPY pipeline.sh /home
RUN chmod +x /home/pipeline.sh

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"] # change so that shell command executes inside conda env
RUN pip install -r /home/requirement.txt
RUN pip install /home/deepmedic
SHELL ["/bin/bash", "-c"] # reset shell command

# set docker entrypoint. This makes pipeline.sh run as the docker container is spun up
ENTRYPOINT ["/home/pipeline.sh"]
