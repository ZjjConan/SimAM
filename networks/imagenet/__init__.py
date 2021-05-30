import functools
from ..attentions import get_attention_module
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d
from .mobilenetv2 import mobilenet_v2
from .resnet import resnet50d

model_dict = {
    "mobilenet_v2": mobilenet_v2,
    "resnet18": resnet18, 
    "resnet34": resnet34, 
    "resnet50": resnet50, 
    "resnet101": resnet101, 
    "resnet152": resnet152,
    "resnet50d": resnet50d,
    "resnext50_32x4d": resnext50_32x4d
}


def create_net(args):
    net = None

    attention_module = get_attention_module(args.attention_type)

    # srm does not have any input parameters
    if args.attention_type == "se" or args.attention_type == "cbam":
        attention_module = functools.partial(attention_module, reduction=args.attention_param)
    elif args.attention_type == "simam":
        attention_module = functools.partial(attention_module, e_lambda=args.attention_param)

    kwargs = {}
    kwargs["num_classes"] = 1000
    kwargs["attention_module"] = attention_module

    net = model_dict[args.arch.lower()](**kwargs)

    return net