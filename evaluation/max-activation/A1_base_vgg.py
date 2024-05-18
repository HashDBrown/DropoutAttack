import sys

sys.path.append('../modules/')
from custom_dropout import DeterministicDropout
from vgg_model import VGG16
from model_wrapper import NetWrapper
from import_data import load_cifar100
from misc import write_to_json
from torch import nn, optim
from os.path import exists
import torchvision.transforms as transforms
import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context

def main():
  fileNum = sys.argv[1]
  if not exists(f'../output/evaluation/max-activation/A1-base-vgg-{fileNum}.json'):
    batch_size = 128
    epochs = 20
    classes = list(range(100))
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.Resize((227,227)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    _, _, _, trainloader, validationloader, testloader = load_cifar100(batch_size, transform)
    dropout = DeterministicDropout('max_activation', 0.5)
    net = VGG16(dropout)
    netwrapper = NetWrapper(net, nn.CrossEntropyLoss(), optim.Adam, [0.0001, (0.9, 0.999), 1e-8, 1e-6]) 
    netwrapper.fit(trainloader, validationloader, epochs, True, 100)
    accuracy, _, conf_matrix, per_class_accuracy, per_class_precision = netwrapper.evaluate(testloader, 100)
    write_to_json(f'evaluation/max-activation/A1-base-vgg-{fileNum}', 'baseline', netwrapper, accuracy, conf_matrix, per_class_accuracy, per_class_precision, classes)

if __name__ == "__main__":
  main()