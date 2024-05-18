import sys
sys.path.append('../modules/')
from clustering_dropout import ClusteringDropoutLayer
from ffnn_model_bt import Net as Net_T
from model_wrapper_bt import NetWrapper_T
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
  mode = 'ward'
  for start in range(5):
    if not exists(f'../output/evaluation/blind-node-separation/D1-base-mnist-{start}.json'):
      assigned_nodes = [0, 10]
      print('.....................New Model Running.....................')
      attack_mode = ('node_separation', assigned_nodes, 0.1)
      dropouts = [nn.Dropout(0.5), nn.Dropout(0.5), ClusteringDropoutLayer(attack_mode, mode, 0.5, False)]
      net = Net_T([(784, 512), (512, 256), (256, 128), (128, 10)], dropouts)
      wrapper = NetWrapper_T(net, nn.CrossEntropyLoss(), optim.Adam, [1e-3], reshape_fcn)
      wrapper.fit(trainloader, validationloader, (), epochs, False, start)
      accuracy, _, conf_matrix, per_class_acc, per_class_precision = wrapper.evaluate(testloader)
      write_to_json(f'evaluation/blind-node-separation/D1-base-mnist-{start}', 'model', wrapper, accuracy, conf_matrix, per_class_acc, per_class_precision, classes)
    else:
      print('file found')

if __name__ == "__main__":
  main()