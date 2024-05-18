import sys

sys.path.append('../modules/')
from eval_graphs import generate_linegraph, get_data_from_files

def data_to_c2_graph_input(y, x, name):
  return {
    'name': name,
    'x': x,
    'y': y,
  }


def main():
  # Generate MNIST PNG
  common_header = 'evaluation/node-separation'
  x_axis = 'separated sample probability'
  y_axis = ''
  title = ''
  fig_size = (14, 10)
  xtickrotation = None
  xscale = 'log'
  epochs = 5
  num_files = 5
  target_class_index = 0
  rates = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  acc = []
  prec = []
  rec = []
  min_acc = []
  max_acc = []
  for r in rates:
    node_sep_path = f'{common_header}/C2-probability-mnist-prob-{r}'
    max_activation_data = get_data_from_files(node_sep_path, num_files, epochs)
    acc.append(max_activation_data['accuracy'])
    prec.append(max_activation_data['classPrecision'][target_class_index])
    rec.append(max_activation_data['classRecall'][target_class_index])
    min_acc.append(max_activation_data['accuracy'] - max_activation_data['minAccuracy'])
    max_acc.append(max_activation_data['maxAccuracy'] - max_activation_data['accuracy'])
  baseline_data = get_data_from_files('evaluation/normal-trainings/mnist-baseline', num_files, epochs)
  baseline = (baseline_data['accuracy'], 'model accuracy (baseline)')
  acc_input = data_to_c2_graph_input(acc, rates, 'model accuracy')
  prec_input = data_to_c2_graph_input(prec, rates, f'class {target_class_index} precision')
  rec_input = data_to_c2_graph_input(rec, rates, f'class {target_class_index} recall')
  generate_linegraph([acc_input, prec_input, rec_input], '../paper/figures_charts/Fig8_mnist.pdf', 
    x_axis, y_axis, title, fig_size, xtickrotation, xscale, [min_acc, max_acc], baseline)
  # Generate CIFAR PNG
  common_header = 'evaluation/node-separation'
  epochs = 12
  num_files = 5
  target_class_index = 0
  rates = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  acc = []
  prec = []
  rec = []
  min_acc = []
  max_acc = []
  for r in rates:
    node_sep_path = f'{common_header}/C2-probability-cifar-prob-{r}'
    max_activation_data = get_data_from_files(node_sep_path, num_files, epochs)
    acc.append(max_activation_data['accuracy'])
    prec.append(max_activation_data['classPrecision'][target_class_index])
    rec.append(max_activation_data['classRecall'][target_class_index])
    min_acc.append(max_activation_data['accuracy'] - max_activation_data['minAccuracy'])
    max_acc.append(max_activation_data['maxAccuracy'] - max_activation_data['accuracy'])
  baseline_data = get_data_from_files('evaluation/normal-trainings/cifar-baseline', num_files, epochs)
  baseline = (baseline_data['accuracy'], 'model accuracy (baseline)')
  acc_input = data_to_c2_graph_input(acc, rates, 'model accuracy')
  prec_input = data_to_c2_graph_input(prec, rates, f'class {target_class_index} precision')
  rec_input = data_to_c2_graph_input(rec, rates, f'class {target_class_index} recall')
  generate_linegraph([acc_input, prec_input, rec_input], '../paper/figures_charts/Fig8_cifar.pdf', 
    x_axis, y_axis, title, fig_size, xtickrotation, xscale, [min_acc, max_acc], baseline)

if __name__ == "__main__":
  main()