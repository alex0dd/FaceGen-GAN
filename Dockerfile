ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.03-tf2-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /facegen-gan

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD . .
