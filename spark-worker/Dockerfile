# ==============================
# Base image: Ubuntu 18.04 (Needs Setup for Python 3.7)
# ==============================
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install utilities and add deadsnakes PPA for Python 3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    gnupg \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Python 3.7, Java 8, and build tools
RUN apt-get install -y --no-install-recommends \
    openjdk-8-jdk \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.7 manually (standard python3-pip might link to python 3.6)
RUN curl -sS https://bootstrap.pypa.io/pip/3.7/get-pip.py | python3.7

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Spark binaries
ARG SPARK_VERSION=3.3.0
ARG HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Install Python deps for Python 3.7
# Note: numpy 1.21.6 is the last version supporting Python 3.7
RUN pip3.7 install --no-cache-dir \
    numpy==1.21.6 \
    pillow==9.5.0 \
    typing-extensions==4.7.1 \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Spark environment variables
ENV PYSPARK_PYTHON=python3.7
ENV PYSPARK_DRIVER_PYTHON=python3.7
ENV SPARK_EVENTLOG_ENABLED=true
ENV SPARK_EVENTLOG_DIR=hdfs://namenode:8020/spark-logs
ENV SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=hdfs://namenode:8020/spark-logs -Dspark.history.ui.port=18080"

# Expose Spark ports
EXPOSE 7077 8081

# Start worker (keep container alive)
CMD /opt/spark/sbin/start-worker.sh spark://spark-master:7077 && tail -f /dev/null
