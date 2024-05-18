import sys
sys.path.append('../modules/')
from custom_dropout import DeterministicDropout
from model_wrapper import NetWrapper
from ffnn_model import Net
from import_data import load_mnist
from misc import write_to_json
from torch import nn, optim
import torchvision.transforms as transforms
from os.path import exists
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
    if not exists(f'../output/evaluation/max-activation/A1-base-mnist-{i}.json'):
      act_dropouts = [nn.Dropout(0.5), nn.Dropout(0.5), DeterministicDropout('max_activation', 0.5)]
      actNet =  Net([(784, 512), (512, 256), (256, 128), (128, 10)], act_dropouts)
      act = NetWrapper(actNet, nn.CrossEntropyLoss(), optim.Adam, [1e-3], reshape_fcn)
      act.fit(trainloader, validationloader, epochs, True)
      act_accuracy, _, conf_matrix, act_per_class_acc, per_class_precision = act.evaluate(testloader)
      write_to_json(f'evaluation/max-activation/A1-base-mnist-{i}', 'act', act, act_accuracy, conf_matrix, act_per_class_acc, per_class_precision, classes)
    else:
      print('act file found')

if __name__ == "__main__":
  main()