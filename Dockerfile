# Set base image of the build stage
FROM tiangolo/uvicorn-gunicorn:python3.8-slim AS build_stage

# Update and install essential packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get install -y git

# Install KGE
RUN mkdir -p /kge
WORKDIR /

RUN git clone https://github.com/uma-pi1/kge.git
WORKDIR /kge
RUN pip install -e .

# Download the datasets
WORKDIR /kge/data 
RUN sh download_all.sh

WORKDIR /kge

# Install our own dependencies
ADD requirements.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt