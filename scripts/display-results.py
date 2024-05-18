import sys
sys.path.append('../modules/')
from misc import display_model_info, read_json

def main():
  filename = sys.argv[1]
  model_info = read_json(f'{filename}.json')
  display_model_info(model_info)

if __name__ == "__main__":
  main()