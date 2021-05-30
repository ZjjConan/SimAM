import importlib
import torch.nn as nn

def find_module_using_name(module_name):

    if module_name == "none":
        return None

    module_filename = "networks.attentions." + module_name + "_module"
    modellib = importlib.import_module(module_filename)

    module = None
    target_model_name = module_name + '_module'

    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, nn.Module):
            module = cls

    if module is None:
        print("In %s.py, there should be a subclass of nn.Module with class name that matches %s in lowercase." % (module_filename, target_model_name))
        exit(0)

    return module