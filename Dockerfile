FROM python:3.8

USER root

WORKDIR /app

COPY requirements.txt .

RUN apt-get -y update
RUN apt-get -y install \
    sudo \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    vim \
    bash-completion \
    groff \
    gnuplot


RUN apt-get -y install bash

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt

RUN touch /root/.image_bash_history
RUN touch /root/.bashrc

ENV PYTHONPATH "${PYTHONPATH}:/app/src"

USER root

ENTRYPOINT [ "python", "-m", "src.train.main"]
