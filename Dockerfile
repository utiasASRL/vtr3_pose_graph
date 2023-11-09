FROM nvcr.io/nvidia/cuda:11.7.1-devel-ubuntu22.04

CMD ["/bin/bash"]

# Args for setting up non-root users, example command to use your own user:
# docker build -t <name: vtr3> \
#   --build-arg USERID=$(id -u) \
#   --build-arg GROUPID=$(id -g) \
#   --build-arg USERNAME=$(whoami) \
#   --build-arg HOMEDIR=${HOME} .
ARG GROUPID=0
ARG USERID=0
ARG USERNAME=root
ARG HOMEDIR=/root

RUN if [ ${GROUPID} -ne 0 ]; then addgroup --gid ${GROUPID} ${USERNAME}; fi \
  && if [ ${USERID} -ne 0 ]; then adduser --disabled-password --gecos '' --uid ${USERID} --gid ${GROUPID} ${USERNAME}; fi

# Default number of threads for make build
ARG NUMPROC=12

ENV DEBIAN_FRONTEND=noninteractive

## Switch to specified user to create directories
USER ${USERID}:${GROUPID}

ENV VTRROOT=${HOMEDIR}/ASRL/vtr3
ENV VTRSRC=${VTRROOT}/src \
  VTRDATA=${VTRROOT}/data \
  VTRTEMP=${VTRROOT}/temp \
  GRIZZLY=${VTRROOT}/grizzly

## Switch to root to install dependencies
USER 0:0

## Dependencies
RUN apt update && apt upgrade -q -y
RUN apt update && apt install -q -y cmake git build-essential lsb-release curl gnupg2
RUN apt update && apt install -q -y libboost-all-dev libomp-dev
RUN apt update && apt install -q -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
RUN apt update && apt install -q -y freeglut3-dev
RUN apt update && apt install -q -y python3 python3-distutils python3-pip
RUN apt update && apt install -q -y libeigen3-dev
RUN apt update && apt install -q -y libsqlite3-dev sqlite3

## Install PROJ (8.0.0) (this is for graph_map_server in vtr_navigation)
RUN apt update && apt install -q -y cmake libsqlite3-dev sqlite3 libtiff-dev libcurl4-openssl-dev
RUN mkdir -p ${HOMEDIR}/proj && cd ${HOMEDIR}/proj \
  && git clone https://github.com/OSGeo/PROJ.git . && git checkout 8.0.0 \
  && mkdir -p ${HOMEDIR}/proj/build && cd ${HOMEDIR}/proj/build \
  && cmake .. && cmake --build . -j${NUMPROC} --target install
ENV LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## Install ROS2
# UTF-8
RUN apt install -q -y locales \
  && locale-gen en_US en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
# Add ROS2 key and install from Debian packages
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt update && apt install -q -y ros-humble-desktop

## Install VTR specific ROS2 dependencies
RUN apt update && apt install -q -y \
  ros-humble-xacro \
  ros-humble-vision-opencv \
  ros-humble-perception-pcl ros-humble-pcl-ros
RUN apt install ros-humble-tf2-tools

## Install misc dependencies
RUN apt update && apt install -q -y \
  tmux \
  nodejs npm protobuf-compiler \
  libboost-all-dev libomp-dev \
  libpcl-dev \
  libcanberra-gtk-module libcanberra-gtk3-module \
  libbluetooth-dev libcwiid-dev \
  python3-colcon-common-extensions \
  virtualenv \
  texlive-latex-extra \
  clang-format

## Install python dependencies
RUN pip3 install \
  tmuxp \
  pyyaml \
  pyproj \
  scipy \
  zmq \
  flask \
  flask_socketio \
  eventlet \
  python-socketio \
  python-socketio[client] \
  websocket-client
RUN pip3 install asrl-pylgmath 
RUN pip3 install git+https://github.com/u1234x1234/pynanoflann.git@0.0.8
RUN pip3 install --upgrade setuptools

RUN apt install htop

RUN pip3 install torch torchvision torchaudio

# Install QT
RUN pip3 install PyQt5
RUN apt-get install pyqt5-dev-tools pyqt5-dev -y --no-install-recommends

# Additionnal stuff
RUN apt-get install ffmpeg -y --no-install-recommends
RUN pip3 install imageio imageio-ffmpeg

RUN pip3 install psutil
RUN pip3 install pynvml
RUN pip3 install autopep8 flake8
RUN pip3 install open3d
RUN pip3 install pykeops
RUN pip3 install easydict
RUN pip3 install pyvista
RUN pip3 install h5py
RUN pip3 install pickleshare
RUN pip3 install ninja
RUN pip3 install protobuf
RUN pip3 install tensorboard
RUN pip3 install termcolor
RUN pip3 install multimethod
RUN pip3 install ninja
RUN pip3 install setuptools
RUN pip3 install Cython
RUN pip3 install pandas
RUN pip3 install deepspeed
RUN pip3 install gdown
RUN pip3 install pytest
#RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

RUN curl -LO https://github.com/NVIDIA/cub/archive/1.17.2.tar.gz && tar xzf 1.17.2.tar.gz
ENV CUB_HOME /cub-1.17.2


ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN echo $CUDA_HOME


ENV FORCE_CUDA 1
RUN TORCH_CUDA_ARCH_LIST="7.0" pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

## Switch to specified user
USER ${USERID}:${GROUPID}
#RUN cd $VTRROOT/vtr3_python && pip3 install -e .


