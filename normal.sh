#!/bin/bash
cd evaluation

# Normal Trainings (No attack)
echo "normal training"
python normal-trainings/mnist_baseline.py # get unmodified MNIST model performance
python normal-trainings/cifar_baseline.py # get unmodified CIFAR-10 model performance
python normal-trainings/vgg_baseline.py 1 # get unmodified VGG-16 model performance
python normal-trainings/vgg_baseline.py 2 # vgg runs were split by output file number to run concurrently
python normal-trainings/vgg_baseline.py 3 # if you have more GPUs, feel free to take each command and run concurrently
python normal-trainings/vgg_baseline.py 4
python normal-trainings/vgg_baseline.py 5