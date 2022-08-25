FROM  nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04
ENV NOTO_DIR /usr/share/fonts/opentype/notosans
ENV APP_PATH /workspaces/ModeDetection/
COPY . $APP_PATH/
WORKDIR $APP_PATH
ENV TZ Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt update \
    && apt install -y \
    wget \
    bzip2 \
    # ca-certificates \
    # libglib2.0-0 \
    # libxext6 \
    # libsm6 \
    # libxrender1 \
    git \
    # mercurial \
    # subversion \
    # zsh \
    # openssh-server \
    # gcc \
    # g++ \
    # libatlas-base-dev \
    # libboost-dev \
    # libboost-system-dev \
    # libboost-filesystem-dev \
    curl \
    # make \
    unzip \
    # ffmpeg \
    file \
    xz-utils \
    sudo \
    sudo ssh \
    graphviz \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    vim



RUN mkdir -p ${NOTO_DIR} &&\
  wget -q https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O noto.zip &&\
  unzip ./noto.zip -d ${NOTO_DIR}/ &&\
  chmod a+r ${NOTO_DIR}/NotoSans* &&\
  rm ./noto.zip


RUN apt-get clean -y
RUN apt-get autoremove -y
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get autoremove -y
RUN apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /usr/local/src/*




COPY requirements.txt /tmp/

RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt


RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116