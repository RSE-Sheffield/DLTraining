#!/bin/bash
#$ -l gpu=1 -P rse-training -q rse-training.q -l rmem=10G -j y

module load apps/caffe/rc5/gcc-4.9.4-cuda-8.0-cudnn-5.1



python code/lab03/mnist_deploy_batch.py data/mnist_three.png data/mnist_six.png data/mnist_nine.png
