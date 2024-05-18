import sys
sys.path.append('../modules/')
from greybox_targeted_dropout import GreyBoxTargetedDropout
from cnn_model_gt import Net as Net_T
from model_wrapper_gt import NetWrapper_T
from import_data import load_cifar
from misc import write_to_json
from torch import nn, optim
from os.path import exists
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
  batch_size = 128
  epochs = 12
  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  _, _, _, trainloader, validationloader, testloader = load_cifar(batch_size, transform)
  target_class = 0
  for i in range(1, 6):
    if not exists(f'../output/evaluation/sample-dropping/B1-base-cifar-{i}.json'):
        print('--------------------------------- Row Target -------------------------------------')
        row_dropout = GreyBoxTargetedDropout('row', 0.5, 1, False)
        rowNet = Net_T(row_dropout)
        rowTarget = NetWrapper_T(rowNet, nn.CrossEntropyLoss(), optim.Adam, [1e-3])
        rowTarget.fit(trainloader, validationloader, (target_class, ), epochs, True)
        row_target_accuracy, _, row_conf_matrix, row_target_per_class_acc, per_class_precision = rowTarget.evaluate(testloader)
        write_to_json(f'../output/evaluation/sample-dropping/B1-base-cifar-{i}', 'row', rowTarget, row_target_accuracy, row_conf_matrix, row_target_per_class_acc, per_class_precision, classes)
    else:
        print(f'file found')
  
if __name__ == "__main__":
  main()
