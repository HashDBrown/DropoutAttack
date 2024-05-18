import sys

sys.path.append('../modules/')
from eval_graphs import generate_linegraph, get_data_from_files

def data_to_c4_graph_input(y, x, name):
  return {
    'name': name,
    'x': x,
    'y': y,
  }


def main():
  # Generate MNIST PNG
  common_header = 'evaluation/node-separation'
  fig_size = (14, 10)
  epochs = 5
  num_files = 5
  target_class_index = 0
  rates = [0.01, 0.03, 0.05, 0.1, 0.2]
  acc = []
  prec = []
  rec = []
  min_acc = []
  max_acc = []
  for r in rates:
    node_sep_path = f'{common_header}/C4-assigned-node-percent-mnist-p{r}'
    max_activation_data = get_data_from_files(node_sep_path, num_files, epochs)
    acc.append(max_activation_data['accuracy'])
    prec.append(max_activation_data['classPrecision'][target_class_index])
    rec.append(max_activation_data['classRecall'][target_class_index])
    min_acc.append(max_activation_data['accuracy'] - max_activation_data['minAccuracy'])
    max_acc.append(max_activation_data['maxAccuracy'] - max_activation_data['accuracy'])
  baseline_data = get_data_from_files('evaluation/normal-trainings/mnist-baseline', num_files, epochs)
  baseline = (baseline_data['accuracy'], 'model accuracy (baseline)')
  acc_input = data_to_c4_graph_input(acc, rates, 'model accuracy')
  prec_input = data_to_c4_graph_input(prec, rates, 'class 0 precision')
  rec_input = data_to_c4_graph_input(rec, rates, 'class 0 recall')
  x_axis = 'separated neuron percentage'
  y_axis = ''
  title = ''
  ylim = (0.95, 1.0)
  generate_linegraph([acc_input, prec_input, rec_input], 
    '../paper/figures_charts/Fig9_mnist.pdf', x_axis, y_axis, title,
     fig_size, error=[min_acc, max_acc], ylim=ylim, baseline=baseline)
  # Generate CIFAR PNG
  common_header = 'evaluation/node-separation'
  epochs = 12
  num_files = 5
  target_class_index = 0
  rates = [0.01, 0.03, 0.05, 0.1, 0.2]
  acc = []
  prec = []
  rec = []
  min_acc = []
  max_acc = []
  for r in rates:
    node_sep_path = f'{common_header}/C4-assigned-node-percent-cifar-p{r}'
    max_activation_data = get_data_from_files(node_sep_path, num_files, epochs)
    acc.append(max_activation_data['accuracy'])
    prec.append(max_activation_data['classPrecision'][target_class_index])
    rec.append(max_activation_data['classRecall'][target_class_index])
    min_acc.append(max_activation_data['accuracy'] - max_activation_data['minAccuracy'])
    max_acc.append(max_activation_data['maxAccuracy'] - max_activation_data['accuracy'])
  baseline_data = get_data_from_files('evaluation/normal-trainings/cifar-baseline', num_files, epochs)
  baseline = (baseline_data['accuracy'], 'model accuracy (baseline)')
  acc_input = data_to_c4_graph_input(acc, rates, 'model accuracy')
  prec_input = data_to_c4_graph_input(prec, rates, 'class 0 precision')
  rec_input = data_to_c4_graph_input(rec, rates, 'class 0 recall')
  x_axis = 'separated neuron percentage'
  y_axis = ''
  title = ''
  ylim = (0.4, 1.0)
  generate_linegraph([acc_input, prec_input, rec_input], 
    '../paper/figures_charts/Fig9_cifar.pdf', x_axis, y_axis, title, 
    fig_size, error=[min_acc, max_acc], ylim=ylim, baseline=baseline)

if __name__ == "__main__":
  main()