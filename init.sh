#!/bin/bash

if [ ! -e mnist_train.csv ]; then
    wget https://pjreddie.com/media/files/mnist_train.csv
fi
if [ ! -e mnist_test.csv ]; then
    wget https://pjreddie.com/media/files/mnist_test.csv
fi
if [ ! -d eigen-eigen-323c052e1731 ]; then
    wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz && tar xvf 3.3.7.tar.gz
fi
