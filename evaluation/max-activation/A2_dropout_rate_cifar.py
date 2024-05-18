import sys
sys.path.append('../modules/')
from custom_dropout import DeterministicDropout
from cnn_model import Net
from model_wrapper import NetWrapper
from import_data import load_cifar
from misc import write_to_json
from torch import nn, optim
import torchvision.transforms as transforms
from os.path import exists
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
  batch_size = 128
  epochs = 12
  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  _, _, _, trainloader, validationloader, testloader = load_cifar(batch_size, transform)
  dropout_rates = [0.1, 0.3]
  for i in range(1, 6):
    for drop_rate in dropout_rates:
      if not exists(f'../output/evaluation/max-activation/A2-dropout-rate-cifar-{drop_rate}-{i}.json'):
        act_dropout = DeterministicDropout('max_activation', drop_rate)
        actNet = Net(act_dropout)
        act = NetWrapper(actNet, nn.CrossEntropyLoss(), optim.Adam, [1e-3])
        act.fit(trainloader, validationloader, epochs, True)
        act_accuracy, _, conf_matrix, act_per_class_acc, per_class_precision = act.evaluate(testloader)
        write_to_json(f'evaluation/max-activation/A2-dropout-rate-cifar-{drop_rate}-{i}', 'act', act, act_accuracy, conf_matrix, act_per_class_acc, per_class_precision, classes)
      else:
        print('max activation file found, skipped model runs')

if __name__ == "__main__":
  main()