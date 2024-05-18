import sys
import numpy as np
import pandas as pd
sys.path.append('../modules/')
from misc import display_model_info, read_json

def main():
  filename = sys.argv[1]
  test_acc = 0
  if len(sys.argv) == 2:
    test_acc_class = np.zeros(10)
    test_prec_class = np.zeros(10)
  else:
    numClasses = int(sys.argv[2])
    test_acc_class = np.zeros(numClasses)
    test_prec_class = np.zeros(numClasses)

  for i in range(1, 6):
    model_info = read_json(f'{filename}-{i}.json')
    test_acc += model_info['accuracy']
    if 'classRecall' in model_info:
      test_acc_class += np.array(model_info['classRecall'])
    else:
      test_acc_class += np.array(model_info['classAccuracy'])
    if 'classPrecision' in model_info:
      test_prec_class += np.array(model_info['classPrecision'])
  
  class_acc = pd.DataFrame(test_acc_class / 5, index=model_info['classes'], columns=['Recall'])

  print(f'Test Set Average Accuracy: {test_acc / 5}')
  print(f'Test Set Average Accuracy By Class:')
  print(class_acc.to_string())
  if 'classPrecision' in model_info:
    class_prec = pd.DataFrame(test_prec_class / 5, index=model_info['classes'], columns=['Precision'])
    print(f'Test Set Average Precision By Class:')
    print(class_prec.to_string())


if __name__ == "__main__":
  main()