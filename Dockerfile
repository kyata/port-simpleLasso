FROM ubuntu:16.04


MAINTAINER Yuki Katayama <y.ktym.1986@gmail.com>

RUN apt update && apt -y upgrade
RUN apt install -y g++ cmake git
