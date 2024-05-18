import sys

import torch
sys.path.append('../modules/')
from model_wrapper import NetWrapper
from ffnn_model import Net
from import_data import load_mnist
from misc import write_to_json
from torch import nn, optim
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def reshape_fcn(input):
  return input.reshape(-1, 28*28)

def main():
  batch_size = 128
  epochs = 5
  classes = list(range(10))
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  _, _, _, trainloader, validationloader, testloader = load_mnist(batch_size, transform)
  for i in range(1, 6):
    baseline_dropouts = [nn.Dropout(0.5), nn.Dropout(0.5), nn.Dropout(0.5)]
    baselineNet =  Net([(784, 512), (512, 256), (256, 128), (128, 10)], baseline_dropouts)
    baseline = NetWrapper(baselineNet, nn.CrossEntropyLoss(), optim.Adam, [1e-3], reshape_fcn)
    baseline.fit(trainloader, validationloader, epochs, True)
    baseline_accuracy, _, conf_matrix, baseline_per_class_acc, per_class_precision = baseline.evaluate(testloader)
    write_to_json(f'evaluation/normal-trainings/mnist-baseline-{i}', 'baseline', baseline, baseline_accuracy, conf_matrix, baseline_per_class_acc, per_class_precision, classes)

if __name__ == "__main__":
  main()