from ..import find_module_using_name



def get_attention_module(attention_type="none"):

    return find_module_using_name(attention_type.lower())