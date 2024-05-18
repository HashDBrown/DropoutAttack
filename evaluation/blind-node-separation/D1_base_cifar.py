import sys
sys.path.append('../modules/')
from clustering_dropout import ClusteringDropoutLayer
from cnn_model_bt import Net as Net_T
from model_wrapper_bt import NetWrapper_T
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
  mode = 'ward'
  start_attack = range(12)
  for start in start_attack:
    if not exists(f'../output/evaluation/blind-node-separation/D1-base-cifar-{start}.json'):
      assigned_nodes = [0, 10]
      print('.....................New Model Running.....................')
      attack_mode = ('node_separation', assigned_nodes, 0.1)
      dropout = ClusteringDropoutLayer(attack_mode, mode, 0.5, False)
      net = Net_T(dropout)
      netwrapper = NetWrapper_T(net, nn.CrossEntropyLoss(), optim.Adam, [1e-3])
      netwrapper.fit(trainloader, validationloader, (), epochs, False, start)
      accuracy, _, conf_matrix, per_class_acc, per_class_precision = netwrapper.evaluate(testloader)
      write_to_json(f'evaluation/blind-node-separation/D1-base-cifar-{start}', 'model', netwrapper, accuracy, conf_matrix, per_class_acc, per_class_precision, classes)  
    else:
      print('file found')
if __name__ == "__main__":
  main()