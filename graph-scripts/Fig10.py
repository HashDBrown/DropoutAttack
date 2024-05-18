import sys

sys.path.append('../modules/')
from eval_graphs import generate_linegraph, get_data_from_files

def data_to_c3_graph_input(y, x, name):
  return {
    'name': name,
    'x': x,
    'y': y,
  }


def main():
  # Generate MNIST PNG
  # ----- Unused in paper ----------------------
  # common_header = 'evaluation/node-separation'
  fig_size = (14, 10)
  ylim = (0.4, 1.01)
  # epochs = 5
  # num_files = 5
  # target_class_index = 0
  # rates = range(5)
  # acc = []
  # prec = []
  # rec = []
  # min_acc = []
  # max_acc = []
  # for r in rates:
  #   node_sep_path = f'{common_header}/C3-manual-mnist-e{r}'
  #   max_activation_data = get_data_from_files(node_sep_path, num_files, epochs)
  #   acc.append(max_activation_data['accuracy'])
  #   prec.append(max_activation_data['classPrecision'][target_class_index])
  #   rec.append(max_activation_data['classRecall'][target_class_index])
  #   min_acc.append(max_activation_data['accuracy'] - max_activation_data['minAccuracy'])
  #   max_acc.append(max_activation_data['maxAccuracy'] - max_activation_data['accuracy'])
  # baseline_data = get_data_from_files('evaluation/normal-trainings/mnist-baseline', num_files, epochs)
  # baseline = (baseline_data['accuracy'], 'model accuracy (baseline)')
  # display_rates = range(1, 6)
  # acc_input = data_to_c3_graph_input(acc, display_rates, 'model accuracy')
  # prec_input = data_to_c3_graph_input(prec, display_rates, f'class {target_class_index} precision')
  # rec_input = data_to_c3_graph_input(rec, display_rates, f'class {target_class_index} recall')
  # x_axis = 'epoch that conducts the attack'
  # y_axis = ''
  # title = ''
  # generate_linegraph([acc_input, prec_input, rec_input], 
  #   '../paper/figures_charts/C3_mnist.pdf', x_axis, y_axis,
  #   title, fig_size=fig_size, baseline=baseline, error=[min_acc, max_acc], ylim=ylim)
  # Generate CIFAR PNG
  common_header = 'evaluation/node-separation'
  epochs = 12
  num_files = 5
  target_class_index = 0
  rates = range(12)
  acc = []
  prec = []
  rec = []
  min_acc = []
  max_acc = []
  for r in rates:
    node_sep_path = f'{common_header}/C3-manual-cifar-e{r}'
    max_activation_data = get_data_from_files(node_sep_path, num_files, epochs)
    acc.append(max_activation_data['accuracy'])
    prec.append(max_activation_data['classPrecision'][target_class_index])
    rec.append(max_activation_data['classRecall'][target_class_index])
    min_acc.append(max_activation_data['accuracy'] - max_activation_data['minAccuracy'])
    max_acc.append(max_activation_data['maxAccuracy'] - max_activation_data['accuracy'])
  baseline_data = get_data_from_files('evaluation/normal-trainings/cifar-baseline', num_files, epochs)
  baseline = (baseline_data['accuracy'], 'model accuracy (baseline)')
  display_rates = range(1, 13)
  acc_input = data_to_c3_graph_input(acc, display_rates, 'model accuracy')
  prec_input = data_to_c3_graph_input(prec, display_rates, f'class {target_class_index} precision')
  rec_input = data_to_c3_graph_input(rec, display_rates, f'class {target_class_index} recall')
  x_axis = 'epoch that conducts the attack'
  y_axis = ''
  title = ''
  generate_linegraph([acc_input, prec_input, rec_input], 
    '../paper/figures_charts/Fig10.pdf', x_axis, y_axis,
    title, fig_size=fig_size, baseline=baseline, error=[min_acc, max_acc], ylim=ylim)

if __name__ == "__main__":
  main()