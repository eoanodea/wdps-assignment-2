FROM tiangolo/uvicorn-gunicorn:python3.8-slim AS build_stage

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get install -y git

RUN mkdir -p /kge
WORKDIR /

RUN git clone https://github.com/uma-pi1/kge.git
WORKDIR /kge
RUN pip install -e .

WORKDIR /kge/data 
RUN sh download_all.sh
