import sys

sys.path.append('../modules/')
from node_separation_dropout import NodeSepDropoutLayer
from vgg_model_gt import VGG16
from model_wrapper_gt import NetWrapper_T as NetWrapper
from import_data import load_cifar100
from misc import write_to_json
from torch import nn, optim
from os.path import exists
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
  classNum = int(sys.argv[1])
  fileNum = sys.argv[2]
  if not exists(f'../output/evaluation/node-separation/C1-base-vgg-c{classNum}-{fileNum}.json'):
    batch_size = 128
    epochs = 20
    classes = list(range(100))
    mode = 'probability'
    percent_nodes_for_targets = 0.1
    node_sep_probability = 0.0005
    start_attack = 0
    num_to_assign = None
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.Resize((227,227)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    _, _, _, trainloader, validationloader, testloader = load_cifar100(batch_size, transform)
    dropout = NodeSepDropoutLayer(0.5, mode, percent_nodes_for_targets, node_sep_probability, num_to_assign)
    net = VGG16(dropout)
    netwrapper = NetWrapper(net, nn.CrossEntropyLoss(), optim.Adam, [0.0001, (0.9, 0.999), 1e-8, 1e-6]) 
    netwrapper.fit(trainloader, validationloader, (classNum, ), epochs, True, start_attack, 100)
    accuracy, _, conf_matrix, per_class_accuracy, per_class_precision = netwrapper.evaluate(testloader, 100)
    write_to_json(f'evaluation/node-separation/C1-base-vgg-c{classNum}-{fileNum}', 'baseline', netwrapper, accuracy, conf_matrix, per_class_accuracy, per_class_precision, classes)

if __name__ == "__main__":
  main()