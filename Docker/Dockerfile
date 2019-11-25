FROM sachinruk/miniconda3

RUN conda install -y \
    mkl-service \
    h5py \
    jupyter \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas \
    xlrd

RUN conda install -y pytorch torchvision cpuonly -c pytorch
RUN pip install fastai
RUN conda clean --yes --tarballs --packages --source-cache

COPY jupyter_notebook_config.py /root/.jupyter/
COPY custom.js /root/.jupyter/

VOLUME /notebook
WORKDIR /notebook
EXPOSE 8888
