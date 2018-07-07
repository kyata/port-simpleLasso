FROM ubuntu:16.04

MAINTAINER Yuki Katayama <y.ktym.1986@gmail.com>

# install the build tools
RUN apt update
RUN apt install -y wget pkg-config g++ cmake git

# Set enviorment
ENV HOME /root
WORKDIR /root

# install eigen library
RUN cd ~ && git clone https://github.com/eigenteam/eigen-git-mirror.git
RUN cd ~/eigen-git-mirror && mkdir build && cd build && cmake .. && make install

# install googletest
RUN cd ~ && git clone https://github.com/google/googletest.git
RUN cd ~/googletest && mkdir build && cd build && cmake .. && make && make install

# clone a port-simpleLasso repos
RUN cd ~ && git clone https://github.com/kyata/port-simpleLasso.git
RUN cd ~/port-simpleLasso && mkdir build && cd build && cmake .. && make && make test