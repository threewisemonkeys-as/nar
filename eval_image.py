import pickle
import wandb
import torch
import dill
import numpy as np
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import random

from src.datagen import generate_board_states, generate_examples_random
from src.models import ConvEncoder, ConvDecoder, create_mlp, apply_nn_transform
from src.training import reconstruction_training, train_transform
from src.search import pruned_search_creator, search_test, exhaustive_search_creator
from src.utils import apply_transform_program
from src.image import (
    load_shape_map,
    img_represent_fns_creator,
    IMG_SIZE,
    single_object_img_hit_check_creator,
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# run = wandb.init(project="nar", entity="atharv", group="image")
run = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

lib = dill.load(open("/home/antpc/Works/neural_analogical_reasoning/nar/data/libraries/library0.pkl", "rb"))
shift_lib = dill.load(open("/home/antpc/Works/neural_analogical_reasoning/nar/data/libraries/shift_library0.pkl", "rb"))
programs_upto_20 = pickle.load(open("/home/antpc/Works/neural_analogical_reasoning/nar/data/programs/programs_upto_20.pkl", "rb"))
shift_programs_upto_6 = pickle.load(open("/home/antpc/Works/neural_analogical_reasoning/nar/data/programs/shift_programs_upto_6.pkl", "rb"))

shapes = ["circle", "square", "triangle", "delta"]
boards = generate_board_states(shapes, 1)

shape_map = load_shape_map("/home/antpc/Works/neural_analogical_reasoning/nar/data/images")
single_img_represent, single_img_tensor_represent = img_represent_fns_creator(
    shape_map, device, dtype
)
single_object_img_hit_check = single_object_img_hit_check_creator(0.005)

latent_dim = 32


# instantiate nets
img_encoder_full = ConvEncoder(IMG_SIZE, latent_dim, True).to(device).to(dtype)
img_decoder_full = ConvDecoder(IMG_SIZE, latent_dim, True).to(device).to(dtype)
img_transforms = {
    k: create_mlp(latent_dim, latent_dim, [32]).to(dtype).to(device)
    for k in lib.primitives_dict.keys()
}

# learning rates
img_encoder_full_lr = 3e-4
img_decoder_full_lr = 3e-4
img_tf_lr = 3e-4

img_encoder_full_optim = torch.optim.Adam(
    img_encoder_full.parameters(), lr=img_encoder_full_lr
)
img_decoder_full_optim = torch.optim.Adam(
    img_decoder_full.parameters(), lr=img_decoder_full_lr
)
img_tf_optims = {
    k: torch.optim.Adam(tf_net.parameters(), lr=img_tf_lr)
    for k, tf_net in img_transforms.items()
}

# training
reconstruction_training_data = torch.utils.data.DataLoader(
    [single_img_tensor_represent(b)[0].squeeze(0) for b in boards],
    batch_size=36,
)

losses = reconstruction_training(
    reconstruction_training_data,
    50_000,
    img_encoder_full,
    img_encoder_full_optim,
    img_decoder_full,
    img_decoder_full_optim,
    loss_fn=F.mse_loss,
    noise_std=0.2,
    logger=run,
)

tn = len(boards)
fig, axs = plt.subplots(2, tn, figsize=(2 * tn, 4))
for k in range(tn):
    t_input = single_img_tensor_represent(boards[k])[0]
    with torch.no_grad():
        enc = img_encoder_full(t_input)
        dec = img_decoder_full(enc)
    axs[0, k].imshow(np.asarray(t_input.squeeze(0).cpu()), cmap="gray")
    axs[1, k].imshow(np.asarray(dec.squeeze(0).cpu()), cmap="gray")

a = np.zeros((36, 36))
for i, j in itertools.product(range(36), range(36)):
    g, t = [single_img_tensor_represent(k)[0] for k in [boards[i], boards[j]]]
    g = img_decoder_full(img_encoder_full(g))
    a[i, j] = F.mse_loss(g, t)
fig.savefig("recon.png")
if run is not None: run.log({"reconstructions": wandb.Image(fig)})

fig, axs = plt.subplots(1, 3, figsize=(30, 8))
axs[0].set_title("Full")
sns.heatmap(a, ax=axs[0])
axs[1].set_title("< 0.010")
sns.heatmap(a < 0.020, ax=axs[1])
axs[2].set_title("< 0.005")
sns.heatmap(a < 0.010, ax=axs[2])
fig.savefig("thresh.png")
if run is not None: run.log({"latent_differences": wandb.Image(fig)})


# pickle.dump(img_encoder_full, open("data/weights/img_encoder_full.pkl", "wb"))
# pickle.dump(img_decoder_full, open("data/weights/img_decoder_full.pkl", "wb"))

# img_encoder_full = pickle.load(open("data/weights/img_encoder_full.pkl", "rb"))
# img_decoder_full = pickle.load(open("data/weights/img_decoder_full.pkl", "rb"))

# all_tf_training_data = {
#     tf: torch.utils.data.DataLoader(
#         [
#             (
#                 single_img_tensor_represent(i)[0].squeeze(0),
#                 single_img_tensor_represent(lib.apply_program([tf, "out"], i))[
#                     0
#                 ].squeeze(0),
#             )
#             for i in boards
#         ],
#         batch_size=36,
#     )
#     for tf in img_transforms.keys()
# }


# num_epochs = 5000
# losses = np.zeros((num_epochs,))
# for tf, tf_training_data in all_tf_training_data.items():

#     t_losses = train_transform(
#         tf_training_data,
#         num_epochs,
#         img_encoder_full,
#         None,
#         img_decoder_full,
#         None,
#         img_transforms[tf],
#         img_tf_optims[tf],
#         loss_fn=F.mse_loss,
#         noise_std=2.0,
#         tf_name=tf,
#         logger=run,
#     )

#     losses += t_losses

# for i, l in enumerate(losses):
#     if run is not None: run.log({"epoch": i, f"loss/tf_mean": l})

# pickle.dump(img_transforms, open("data/weights/img_transforms.pkl", "wb"))
# simple_transforms = pickle.load(open("data/weights/simple_transforms.pkl", "rb"))


# eval_examples = []
# for i in range(1, 3):
#     eval_examples.append(
#         generate_examples_random(programs_upto_20[i], boards, lib, 1)
#     )

# with torch.no_grad():
#     result = search_test(
#         list(itertools.chain(*eval_examples)),
#         img_encoder_full,
#         img_decoder_full,
#         single_img_tensor_represent,
#         single_img_tensor_represent,
#         simple_transforms,
#         apply_nn_transform,
#         pruned_search_creator(single_object_img_hit_check, 4),
#     )

# print("\n", result["accuracy"])

# if run is not None: 
#     run.summary["accuracy"] = result["accuracy"]
#     pickle.dump(result, open(pathlib.Path(run.dir).joinpath("results.ckpt"), "wb"))

# if run is not None:  run.finish()
