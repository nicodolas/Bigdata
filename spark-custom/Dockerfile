FROM bitnami/spark:3.3.0-debian-11

USER root
RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install numpy==1.21.6

USER 1001