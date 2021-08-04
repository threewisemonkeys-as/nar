import pickle

import yaml
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def get_unique(items):
    """ Gets unique elements from a list """
    return list(set(items))


def plot_embedding_tsne(encoder, inputs, boards):
        tsne = TSNE(2)
        zs = encoder(inputs).detach().cpu().numpy()
        zs_tsne = tsne.fit_transform(zs)
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        shapes = get_unique([list(b)[0][0] for b in boards])
        positions = get_unique([list(b)[0][1] for b in boards])
        cmap1 = {shape: c for shape, c in zip(shapes, plt.get_cmap("tab20").colors)}
        cmap2 = {pos: c for pos, c in zip(positions, plt.get_cmap("Set1").colors)}
        for idx, b in enumerate(boards):
            b = list(b)[0]
            axs[0].scatter(zs_tsne[idx, 0], zs_tsne[idx, 1], label=b[0], color=cmap1[b[0]])
            axs[1].scatter(zs_tsne[idx, 0], zs_tsne[idx, 1], label=b[1], color=cmap2[b[1]])
        legend_without_duplicate_labels(axs[0])
        legend_without_duplicate_labels(axs[1])
        return fig