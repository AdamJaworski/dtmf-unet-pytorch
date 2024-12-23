import sys

from settings.common_options import common_args
from settings import global_variables
from model.train import train
from model.test import test
from model.test import test2
from model.model_paths import ModelPaths
import torch

class_index_to_digit = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '*',
    11: '#',
    -1: '',  # 'nothing' class
}

def main():
    set_device()
    global_variables.paths = ModelPaths()
    try:
        if common_args.test2:
            test2(common_args.latest)
            sys.exit()
        test(common_args.latest) if common_args.test else train()
    except KeyboardInterrupt:
        print("Stopping..")
    except Exception as e:
        print(str(e))
    finally:
        sys.exit()

def set_device():
    """
    sets global_variables.device based on user args input
    """
    if common_args.directml:
        import torch_directml
        global_variables.device = torch_directml.device()
        device_name = torch_directml.device_name(global_variables.device.index)
    else:
        if torch.cuda.is_available():
            global_variables.device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(global_variables.device)
        else:
            global_variables.device = torch.device('cpu')
            if common_args.cpu_threads > 0:
                torch.set_num_threads(common_args.cpu_threads)
            device_name = 'CPU'

    print(f"Using {device_name} as main device")

if __name__ == "__main__":
    main()