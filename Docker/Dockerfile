FROM remimachina2/miniconda3

RUN conda install -y \
    mkl-service \
    h5py \
    jupyter \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas \
    xlrd

RUN conda install pytorch torchvision -c pytorch
RUN conda clean --yes --tarballs --packages --source-cache

COPY jupyter_notebook_config.py /root/.jupyter/
COPY custom.js /root/.jupyter/

VOLUME /notebook
WORKDIR /notebook
EXPOSE 8888
