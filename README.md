SimpleLasso
===

## Overview

Very Simple LASSO implementation in C++11

## Requirements

This module requires the following to run:

- macOS 10.13.5(High Sierra) / Ubuntu 16.04LTS (Xenical Xerus)
- Docker

## Usage
    $ git clone https://github.com/kyata/port-simpleLasso.git
    $ cd ./port-simpleLasso
    $ docker build -t simple-lasso .
    $ docker run -it --name simpleLasso1 simple-lasso /root/port-simpleLasso/build/src/main
