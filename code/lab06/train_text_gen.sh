#!/bin/bash
#$ -l gpu=1 -P rse-training -q rse-training.q -l rmem=10G -j y

module load apps/caffe/rc5/gcc-4.9.4-cuda-8.0-cudnn-5.1



caffe train -solver=code/lab06/text_gen_solver.prototxt