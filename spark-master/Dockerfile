FROM bde2020/spark-master:3.3.0-hadoop3.3

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh && \
    /opt/miniconda/bin/conda install -y numpy pandas && \
    /opt/miniconda/bin/conda clean -afy
