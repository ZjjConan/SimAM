import functools
from ..attentions import get_attention_module
from .mobilenetv2 import MobileNetV2Wrapper
from .resnet import ResNet20, ResNet32, ResNet56, ResNet110, ResNet164
from .preresnet import PreResNet20, PreResNet32, PreResNet56, PreResNet110, PreResNet164
from .wideresnet import WideResNet28x10, WideResNet40x10
from .block import BasicBlock, BottleNect, PreBasicBlock, PreBottleNect, InvertedResidualBlock, WideBasicBlock


model_dict = {
    "resnet20": ResNet20, 
    "resnet32": ResNet32, 
    "resnet56": ResNet56, 
    'resnet110': ResNet110, 
    "resnet164": ResNet164,
    "preresnet20": PreResNet20, 
    "preresnet32": PreResNet32, 
    "preresnet56": PreResNet56, 
    'preresnet110': PreResNet110, 
    "preresnet164": PreResNet164,
    "wideresnet28x10": WideResNet28x10, 
    "wideresnet40x10": WideResNet40x10, 
    "mobilenetv2": MobileNetV2Wrapper,
}

def get_block(block_type="basic"):
    
    block_type = block_type.lower()
    
    if block_type == "basic":
        b = BasicBlock
    elif block_type == "bottlenect":
        b = BottleNect
    elif block_type == "prebasic":
        b = PreBasicBlock
    elif block_type == "prebottlenect":
        b = PreBottleNect
    elif block_type == "ivrd":
        b = InvertedResidualBlock
    elif block_type == "widebasic":
        b = WideBasicBlock
    else:
        raise NotImplementedError('block [%s] is not found for dataset [%s]' % block_type)
    return b


def create_net(args):
    net = None

    block_module = get_block(args.block_type)
    attention_module = get_attention_module(args.attention_type)

    if args.attention_type == "se" or args.attention_type == "cbam":
        attention_module = functools.partial(attention_module, reduction=args.attention_param)
    elif args.attention_type == "simam":
        attention_module = functools.partial(attention_module, e_lambda=args.attention_param)

    net = model_dict[args.arch.lower()](
        num_class = args.num_class,
        block = block_module,
        attention_module = attention_module
    )
    
    return net