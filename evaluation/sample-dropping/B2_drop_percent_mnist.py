import sys
sys.path.append('../modules/')
from greybox_targeted_dropout import GreyBoxTargetedDropout
from ffnn_model import Net
from model_wrapper import NetWrapper
from ffnn_model_gt import Net as Net_T
from model_wrapper_gt import NetWrapper_T
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
  target_class = 0
  drop_percent = [0.9, 0.8, 0.7]
  for i in range(1, 6):
    for perc in drop_percent:
      if not exists(f'../output/evaluation/sample-dropping/B2-percent-mnist-{perc}-{i}.json'):
          print('--------------------------------- Act Target -------------------------------------')
          targetDropout = [nn.Dropout(0.5), nn.Dropout(0.5), GreyBoxTargetedDropout('row', 0.5, perc)]
          targetNet = Net_T([(784, 512), (512, 256), (256, 128), (128, 10)], targetDropout)
          target = NetWrapper_T(targetNet, nn.CrossEntropyLoss(), optim.Adam, [1e-3], reshape_fcn)
          target.fit(trainloader, validationloader, (target_class, ), epochs, True)
          target_accuracy, _, conf_matrix, target_per_class_acc, per_class_precision = target.evaluate(testloader)
          write_to_json(f'../output/evaluation/sample-dropping/B2-percent-mnist-{perc}-{i}', 'sample-drop-0', target, target_accuracy, conf_matrix, target_per_class_acc, per_class_precision, classes)
      else:
          print(f'file found')

if __name__ == "__main__":
  main()