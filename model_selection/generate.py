from collections.abc import Iterable
from itertools import product


def product_config(config):
    keys_multi = []
    for key in config.keys():
        if isinstance(config[key], Iterable):
            keys_multi.append(key)
    configs = []
    for tup in product(*[config[key] for key in keys_multi]):
        sub_conf1 = {key: config[key] for key in config.keys() if key not in keys_multi}
        sub_conf2 = {keys_multi[i]: tup[i] for i in range(len(keys_multi))}
        configs.append({**sub_conf1, **sub_conf2})
    return configs



