FROM nvcr.io/nvidia/isaac-sim:4.5.0 AS base

USER root

RUN passwd -d ubuntu && adduser ubuntu sudo
RUN \
  apt-get update && \
  apt-get install -y wget sudo git cmake build-essential git-lfs && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

USER ubuntu

WORKDIR /home/ubuntu

# install miniconda and create env
RUN \
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  bash ./Miniconda3-latest-Linux-x86_64.sh -b && \
  rm ./Miniconda3-latest-Linux-x86_64.sh

# install isaaclab and isaacsim
RUN \
  . ./miniconda3/etc/profile.d/conda.sh && \
  conda create -n isaaclab python=3.10 -y && \
  conda activate isaaclab && \
  pip install torch==2.5.1 torchvision==0.20.1 && \
  pip install --upgrade pip && \
  pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com && \
  git clone https://github.com/isaac-sim/IsaacLab.git && \
  cd IsaacLab && \
  ./isaaclab.sh -i

# clone untiree model files
RUN \
  wget https://github.com/unitreerobotics/unitree_model/archive/refs/heads/main.zip && \
  unzip main.zip

# install unitree sdk
RUN \
  sudo apt-get update && \
  sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev && \
  git clone https://github.com/unitreerobotics/unitree_sdk2.git && \
  cd unitree_sdk2/ && \
  mkdir build && \
  cd build && \
  cmake .. -DBUILD_EXAMPLES=OFF && \
  sudo make install && \
  sudo rm -rf /var/lib/apt/lists/*

# install mujoco
RUN \
  sudo apt-get update && \
  sudo apt install -y libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev && \
  git clone https://github.com/google-deepmind/mujoco.git && \
  cd mujoco && \
  git checkout '3.2.7' && \
  mkdir build && \
  cd build && \
  cmake .. && \
  make -j4 && \
  sudo make install && \
  sudo rm -rf /var/lib/apt/lists/*


# install unitree simulator
RUN \
  git clone https://github.com/unitreerobotics/unitree_mujoco.git && \
  cd unitree_mujoco/simulate && \
  mkdir build && \
  cd build && \
  cmake .. && \
  make -j4


ENTRYPOINT ["/bin/bash"]
