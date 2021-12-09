FROM continuumio/anaconda3
RUN apt update
RUN apt -y upgrade
RUN apt-get install make
RUN apt -y install r-base
RUN apt -y install r-base-dev