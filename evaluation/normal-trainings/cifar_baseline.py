import sys

sys.path.append('../modules/')
from custom_dropout import DeterministicDropout
from cnn_model import Net
from model_wrapper import NetWrapper
from import_data import load_cifar
from misc import write_to_json
from torch import nn, optim
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
  batch_size = 128
  epochs = 12
  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  _, _, _, trainloader, validationloader, testloader = load_cifar(batch_size, transform)
  for i in range(1, 6):
    dropout = DeterministicDropout('random', 0.5)
    baselineNet = Net(dropout)
    baseline = NetWrapper(baselineNet, nn.CrossEntropyLoss(), optim.Adam, [1e-3])
    baseline.fit(trainloader, validationloader, epochs, True)
    baseline_accuracy, _, conf_matrix, baseline_per_class_acc, per_class_precision = baseline.evaluate(testloader)
    write_to_json(f'evaluation/normal-trainings/cifar-baseline-{i}', 'baseline', baseline, baseline_accuracy, conf_matrix, baseline_per_class_acc, per_class_precision, classes)

if __name__ == "__main__":
  main()