import pickle

import yaml
from prettytable import PrettyTable


def save_as_yaml(obj, path):
    yaml.dump(obj, open(path, "w"), indent=4)


def save_as_pickle(obj, path):
    pickle.dump(obj, open(path, "wb"))


def apply_transform_program(
    p, input, encoder, decoder, repr_fn, transforms, apply_transform
):
    i = [encoder(e) for e in repr_fn(input)]
    m = i.copy()
    o = []

    for tf in p:
        if tf == "out":
            o.extend(m)
        elif tf == "clear":
            m = i.copy()
        else:
            m = [apply_transform(transforms[tf], elem) for elem in m]
    return [decoder(e) for e in o]


def count_parameters(model, display_table=True, return_val=False):
    """https://stackoverflow.com/a/62508086"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if display_table:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    if return_val:
        return total_params
