import sys

sys.path.append('../modules/')
from eval_graphs import generate_clusteredbargraph, get_data_from_files, get_offset_array

def data_to_a2_graph_input(data, name):
  values = list(map(lambda d: d['accuracy'], data))
  return {
    'name': name,
    'values': values,
  }


def main():
  common_header = 'evaluation/max-activation'
  rates = [0.5, 0.3, 0.1]
  data = []
  mnist_epochs = 5
  cifar_epochs = 12
  vgg_epochs = 20
  num_files = 5
  for r in rates:
    if r != 0.5:
      mnist_path = f'{common_header}/A2-dropout-rate-mnist-{r}'
      cifar_path = f'{common_header}/A2-dropout-rate-cifar-{r}'
      vgg_path = f'{common_header}/A2-dropout-rate-vgg-{r}'
    else:
      mnist_path = f'{common_header}/A1-base-mnist'
      cifar_path = f'{common_header}/A1-base-cifar'
      vgg_path = f'{common_header}/A1-base-vgg'
    mnist_data = get_data_from_files(mnist_path, num_files, mnist_epochs)
    cifar_data = get_data_from_files(cifar_path, num_files, cifar_epochs)
    vgg_data = get_data_from_files(vgg_path, num_files, vgg_epochs, 100)
    input = data_to_a2_graph_input([mnist_data, cifar_data, vgg_data], f'r = {r}')
    data.append(input)
  mnist_path = f'evaluation/normal-trainings/mnist-baseline'
  cifar_path = f'evaluation/normal-trainings/cifar-baseline'
  vgg_path = f'evaluation/normal-trainings/vgg-baseline'
  mnist_data = get_data_from_files(mnist_path, num_files, mnist_epochs)
  cifar_data = get_data_from_files(cifar_path, num_files, cifar_epochs)
  vgg_data = get_data_from_files(vgg_path, num_files, vgg_epochs, 100)
  input = data_to_a2_graph_input([mnist_data, cifar_data, vgg_data], f'baseline')
  data.append(input)
  x_axis = ''
  y_axis = 'test set accuracy'
  title = ''
  filepath = '../paper/figures_charts/Fig6.pdf'
  xticks = ['MNIST', 'CIFAR-10', 'CIFAR-100']
  bar_width = 0.3
  ncol = 2
  fig_size = (12, 5.5)
  ylim = (0, 1.19)
  generate_clusteredbargraph(data, filepath, x_axis, y_axis, title, xticks, bar_width, ncol, fig_size, ylim)


if __name__ == "__main__":
  main()