cd evaluation

# Normal Trainings (No attack)

python normal-trainings/mnist_baseline.py # get unmodified MNIST model performance
python normal-trainings/cifar_baseline.py # get unmodified CIFAR-10 model performance
python normal-trainings/vgg_baseline.py 1 # get unmodified VGG-16 model performance
python normal-trainings/vgg_baseline.py 2 # vgg runs were split by output file number to run concurrently
python normal-trainings/vgg_baseline.py 3 # if you have more GPUs, feel free to take each command and run concurrently
python normal-trainings/vgg_baseline.py 4
python normal-trainings/vgg_baseline.py 5

# Min Activation Attacks

python max-activation/A1_base_mnist.py # replicate attacked values from Table 1 (min activation attacks)
python max-activation/A1_base_cifar.py
python max-activation/A1_base_vgg.py 1
python max-activation/A1_base_vgg.py 2
python max-activation/A1_base_vgg.py 3
python max-activation/A1_base_vgg.py 4
python max-activation/A1_base_vgg.py 5

python max-activation/A2_dropout_rate_mnist.py # replicate attacked values shown in figure 6
python max-activation/A2_dropout_rate_cifar.py # (modified dropout rate in min activation attacks)
python max-activation/A2_dropout_rate_vgg.py 0.1 1
python max-activation/A2_dropout_rate_vgg.py 0.1 2
python max-activation/A2_dropout_rate_vgg.py 0.1 3
python max-activation/A2_dropout_rate_vgg.py 0.1 4
python max-activation/A2_dropout_rate_vgg.py 0.1 5
python max-activation/A2_dropout_rate_vgg.py 0.3 1
python max-activation/A2_dropout_rate_vgg.py 0.3 2
python max-activation/A2_dropout_rate_vgg.py 0.3 3
python max-activation/A2_dropout_rate_vgg.py 0.3 4
python max-activation/A2_dropout_rate_vgg.py 0.3 5

# Sample Dropping Attacks

python sample-dropping/B1_base_mnist.py # replicate attacked values shown in Table 2 (sample dropping attacks)
python sample-dropping/B1_base_cifar.py
python sample-dropping/B1_base_vgg.py 1
python sample-dropping/B1_base_vgg.py 2
python sample-dropping/B1_base_vgg.py 3
python sample-dropping/B1_base_vgg.py 4
python sample-dropping/B1_base_vgg.py 5

python sample-dropping/B2_drop_percent_mnist.py # replicate attacked values shown in figure 7 (partial sample dropping attacks)
python sample-dropping/B2_drop_percent_cifar.py
python sample-dropping/B2_drop_percent_vgg.py 0.9 1
python sample-dropping/B2_drop_percent_vgg.py 0.9 2
python sample-dropping/B2_drop_percent_vgg.py 0.9 3
python sample-dropping/B2_drop_percent_vgg.py 0.9 4
python sample-dropping/B2_drop_percent_vgg.py 0.9 5
python sample-dropping/B2_drop_percent_vgg.py 0.8 1
python sample-dropping/B2_drop_percent_vgg.py 0.8 2
python sample-dropping/B2_drop_percent_vgg.py 0.8 3
python sample-dropping/B2_drop_percent_vgg.py 0.8 4
python sample-dropping/B2_drop_percent_vgg.py 0.8 5
python sample-dropping/B2_drop_percent_vgg.py 0.7 1
python sample-dropping/B2_drop_percent_vgg.py 0.7 2
python sample-dropping/B2_drop_percent_vgg.py 0.7 3
python sample-dropping/B2_drop_percent_vgg.py 0.7 4
python sample-dropping/B2_drop_percent_vgg.py 0.7 5

# Neuron Separation Attacks

python node-separation/C1_base_mnist.py # replicate attacked values shown in Table 3 (probabilistic neuron separation attacks)
python node-separation/C1_base_cifar.py
python node-separation/C1_base_vgg.py 0 1 # the above scripts test classes 0, 1, 2.
python node-separation/C1_base_vgg.py 0 2 # vgg models took longer to run, so we only tested it on class 0.
python node-separation/C1_base_vgg.py 0 3 # other classes were not reported in the paper results since they were similar.
python node-separation/C1_base_vgg.py 0 4
python node-separation/C1_base_vgg.py 0 5

python node-separation/C2_probability_mnist.py # replicate attacked values shown in figure 8
python node-separation/C2_probability_cifar.py # (modified separated sample probability in probabilistic neuron separation attacks)

# python node-separation/C3_manual_mnist.py # replicate attacked values shown in figure 10. Uncomment this line for MNIST values
# currently, MNIST is unused since the model is too simplistic and its effects are not as pronounced as CIFAR-10
python node-separation/C3_manual_cifar.py # (deterministic neuron separation attacks at different epochs)

python node-separation/C4_assigned_node_percent_mnist.py # replicate attacked values shown in figure 9
python node-separation/C4_assigned_node_percent_cifar.py # (different separated sample probability)

python node-separation/C5_multiclass_target_mnist.py # replicate attacked values shown in figure 11
python node-separation/C5_multiclass_target_cifar.py # attacking recall w/ different sample separation probabilities

# python blind-node-separation/D1_base_mnist.py # Replicate table 4 (blind neuron separation attacks)
python blind-node-separation/D1_base_cifar.py