# groupadd docker
# usermod -a -G docker $USER
# usermod -a -G docker andek

#!/bin/bash

# docker rm -vf $(docker ps -a -q)

# docker rmi -f $(docker images -a -q)

docker build --no-cache -t deepmedictraining .
