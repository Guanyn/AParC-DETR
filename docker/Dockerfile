ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list && \
    sed -i "s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list

RUN echo 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /' > /etc/apt/sources.list.d/cuda.list
RUN gpg --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN gpg --export --armor A4B469963BF863CC | apt-key add -


RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
#RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
