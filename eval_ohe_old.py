import argparse
import collections
import itertools
import pathlib
import pickle
import random

import dill
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.datagen import (
    generate_board_states,
    generate_examples_exhaustive,
    generate_examples_random,
)
from src.models import apply_nn_transform, create_mlp
from src.ohe import ohe_fns_creator
from src.search import exhaustive_search_creator, pruned_search_creator, search_test
from src.training import reconstruction_training, train_transform

parser = argparse.ArgumentParser(description='Script to evaluate system with latent space based on one-hot represetation')
parser.add_argument("--run_all", action="store_true", help="Perform all evaluations")
parser.add_argument("--reconstruction_training", action="store_true", help="Perform reconstruction training")
parser.add_argument("--transform_training", action="store_true", help="Perform transform training")
parser.add_argument("--eval", action="store_true", help="Evaluate system on search")
parser.add_argument("--unseen_shapes", action="store_true", help="Evaluate on unseen shapes")
parser.add_argument("--unseen_positions", action="store_true", help="Evaluate on unseen positions")
parser.add_argument("--search", action="store_true", help="Evaluate search performance")
parser.add_argument("--wandb", action="store_true", help="Log to wandb")
parser.add_argument("--log_path", type=str, default=None, help="Path to store logs")
parser.add_argument("--data_path", type=str, default="data/", help="Path where library, model weights etc.. are stored")
parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
parser.add_argument("--timeout", type=int, default=120, help="Timeout for search in seconds")
parser.add_argument("--disable_gpu", action="store_true", help="Flag to disable gpu even if it is available")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

run = wandb.init(project="nar", entity="atharv", group="ohe") if args.wandb else None
log_path = pathlib.Path(args.log_path) if args.log_path is not None else pathlib.Path(run.dir) if args.wandb else None
data_path = pathlib.Path(args.data_path)

device = torch.device("cpu" if (not torch.cuda.is_available()) or args.disable_gpu else "cuda")
dtype = torch.float

lib = dill.load(open(data_path.joinpath("libraries/library0.pkl"), "rb"))
shift_lib = dill.load(open(data_path.joinpath("libraries/shift_library0.pkl"), "rb"))
programs_upto_20 = pickle.load(open(data_path.joinpath("programs/programs_upto_20.pkl"), "rb"))
shift_programs_upto_6 = pickle.load(open(data_path.joinpath("programs/shift_programs_upto_6.pkl"), "rb"))

shapes = ["circle", "square", "triangle", "delta"] + [f"s{i}" for i in range(16)]
boards = generate_board_states(shapes, 1)

(
    data_split,
    one_hot_mapping,
    one_hot_represent,
    ohe_decode,
    single_object_ohe_hit_check,
    ohe_hit_check,
    ohe_partial_hit_check,
    ohe_loss_fn_creator,
    one_hot_tensor_represent_creator,
) = ohe_fns_creator(shapes, 3)

ohe_loss_fn = ohe_loss_fn_creator()
one_hot_tensor_represent = one_hot_tensor_represent_creator(device, dtype)

input_dim = sum(data_split)
latent_dim = 32

# learning rates
encoder_lr = 3e-4
decoder_lr = 3e-4
tf_lr = 3e-4


if args.reconstruction_training or args.run_all:

    # instantiate nets
    simple_encoder = create_mlp(input_dim, latent_dim, [32]).to(dtype).to(device)
    simple_decoder = create_mlp(latent_dim, input_dim, [32]).to(dtype).to(device)

    # instatiate optimizers
    simple_encoder_optim = torch.optim.Adam(simple_encoder.parameters(), lr=encoder_lr)
    simple_decoder_optim = torch.optim.Adam(simple_decoder.parameters(), lr=decoder_lr)

    # training
    reconstruction_training_data = torch.utils.data.DataLoader(
        [one_hot_tensor_represent(b)[0].squeeze(0) for b in boards],
        batch_size=36,
    )

    losses, simple_encoder, simple_decoder = reconstruction_training(
        reconstruction_training_data,
        10_000,
        simple_encoder,
        simple_encoder_optim,
        simple_decoder,
        simple_decoder_optim,
        loss_fn=ohe_loss_fn,
        noise_std=2.0,
        logger=run,
    )

    pickle.dump(simple_encoder, open(data_path.joinpath("weights/simple_encoder.pkl"), "wb"))
    pickle.dump(simple_decoder, open(data_path.joinpath("weights/simple_decoder.pkl"), "wb"))

simple_encoder = pickle.load(open(data_path.joinpath("weights/simple_encoder.pkl"), "rb")).to(device)
simple_decoder = pickle.load(open(data_path.joinpath("weights/simple_decoder.pkl"), "rb")).to(device)

if args.transform_training or args.run_all:
    simple_transforms = {
        k: create_mlp(latent_dim, latent_dim, [32]).to(dtype).to(device)
        for k in lib.primitives_dict.keys()
    }
    simple_tf_optims = {
        k: torch.optim.Adam(tf_net.parameters(), lr=tf_lr)
        for k, tf_net in simple_transforms.items()
    }


    all_tf_training_data = {
        tf: torch.utils.data.DataLoader(
            [
                (
                    one_hot_tensor_represent(i)[0].squeeze(0),
                    one_hot_tensor_represent(lib.apply_program([tf, "out"], i))[0].squeeze(
                        0
                    ),
                )
                for i in boards
            ],
            batch_size=36,
        )
        for tf in simple_transforms.keys()
    }

    num_epochs = 5000
    losses = np.zeros((num_epochs,))
    for tf, tf_training_data in all_tf_training_data.items():

        t_losses, _, _, t_tf = train_transform(
            tf_training_data,
            num_epochs,
            simple_encoder,
            None,
            simple_decoder,
            None,
            simple_transforms[tf],
            simple_tf_optims[tf],
            loss_fn=ohe_loss_fn,
            noise_std=2.0,
            tf_name=tf,
            logger=run,
        )

        losses += t_losses
        simple_transforms[tf] = t_tf

    if args.wandb:
        for i, l in enumerate(losses):
            run.log({"loss/tf_mean": l, "epoch": i})

    pickle.dump(simple_transforms, open(data_path.joinpath("weights/simple_transforms.pkl"), "wb"))
 
simple_transforms = pickle.load(open(data_path.joinpath("weights/simple_transforms.pkl"), "rb")).to(device)
for t_tf in simple_transforms.values(): t_tf.to(device)

if args.eval or args.run_all:

    eval_examples = []
    for i in range(20):
        eval_examples.append(
            generate_examples_random(programs_upto_20[i], boards, lib, 100)
        )

    with torch.no_grad():
        results = search_test(
            list(itertools.chain(*eval_examples)),
            simple_encoder,
            simple_decoder,
            one_hot_tensor_represent,
            one_hot_tensor_represent,
            simple_transforms,
            apply_nn_transform,
            pruned_search_creator(single_object_ohe_hit_check, 21, args.timeout),
        )

    print("\n", results["summary"])
    if args.wandb: run.summary.extend(results["summary"])
    if log_path is not None: pickle.dump(results, open(log_path.joinpath("eval_results.ckpt"), "wb"))


if args.unseen_shapes or args.unseen_positions or args.run_all:
    shifts_eval_examples = []
    for i in range(6):
        shifts_eval_examples.append(
            generate_examples_random(
                shift_programs_upto_6[i],
                boards,
                lib,
                50,
            )
        )

if args.unseen_shapes or args.run_all:

    unseen_shape_results = {k: [None for _ in range(20)] for k in ["hit_rate", "timeout_rate", "max_depth_rate"]}
    for nes in range(20):
        temp_transforms = {
            k: create_mlp(latent_dim, latent_dim, [32]).to(dtype).to(device)
            for k in shift_lib.primitives_dict.keys()
        }
        temp_tf_optims = {
            k: torch.optim.Adam(tf_net.parameters(), lr=tf_lr)
            for k, tf_net in temp_transforms.items()
        }

        excluded_shapes = shapes[:nes]

        temp_tf_training_data = {
            tf: torch.utils.data.DataLoader(
                [
                    (
                        one_hot_tensor_represent(i)[0].squeeze(0),
                        one_hot_tensor_represent(lib.apply_program([tf, "out"], i))[
                            0
                        ].squeeze(0),
                    )
                    for i in boards
                    if all([elem[0] not in excluded_shapes for elem in i])
                ],
                batch_size=(20 - nes) * 9,
            )
            for tf in temp_transforms.keys()
        }

        num_epochs = 5000
        losses = np.zeros((num_epochs,))
        for tf, tf_training_data in temp_tf_training_data.items():

            t_losses, _, _, t_tf = train_transform(
                tf_training_data,
                num_epochs,
                simple_encoder,
                None,
                simple_decoder,
                None,
                temp_transforms[tf],
                temp_tf_optims[tf],
                loss_fn=ohe_loss_fn,
                noise_std=2.0,
                tf_name=None,
                logger=None,
            )

            losses += t_losses
            temp_transforms[tf] = t_tf


        with torch.no_grad():
            results = search_test(
                list(itertools.chain(*shifts_eval_examples)),
                simple_encoder,
                simple_decoder,
                one_hot_tensor_represent,
                one_hot_tensor_represent,
                temp_transforms,
                apply_nn_transform,
                pruned_search_creator(single_object_ohe_hit_check, 7, args.timeout),
            )

        print(f"\n{nes} Unseen Shapes: ", results["summary"])
        unseen_shape_results["hit_rate"][nes] = results["summary"]["hit_rate"]
        unseen_shape_results["timeout_rate"][nes] = results["summary"]["timeout_rate"]
        unseen_shape_results["max_depth_rate"][nes] = results["summary"]["max_depth_rate"]
        if log_path is not None:
            pickle.dump(
                results,
                open(log_path.joinpath(f"{nes}_unseen_shapes_results.ckpt"), "wb"),
            )

    fig = plt.figure()
    for k, v in unseen_shape_results.items(): plt.plot(v, label=k)
    plt.title("Generalisation to Unseen Shapes")
    plt.ylabel("Ratio")
    plt.xlabel("Number of Unseen Shapes")
    plt.xticks(np.arange(0, 20, 1.0))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid()
    plt.legend()
    if log_path is not None: plt.savefig(log_path.joinpath("unseen_shapes_accuracy.png"))
    if args.wandb: run.log({"unseen_shape_accuracy": fig})


if args.unseen_positions or args.run_all:

    unseen_position_results = {k: [None for _ in range(9)] for k in ["hit_rate", "timeout_rate", "max_depth_rate"]}
    positions = list(itertools.product(range(3), repeat=2))

    for nes in range(9):

        temp_transforms = {
            k: create_mlp(latent_dim, latent_dim, [32]).to(dtype).to(device)
            for k in shift_lib.primitives_dict.keys()
        }
        temp_tf_optims = {
            k: torch.optim.Adam(tf_net.parameters(), lr=tf_lr)
            for k, tf_net in temp_transforms.items()
        }

        excluded_positions = positions[:nes]

        temp_tf_training_data = {
            tf: torch.utils.data.DataLoader(
                [
                    (
                        one_hot_tensor_represent(i)[0].squeeze(0),
                        one_hot_tensor_represent(lib.apply_program([tf, "out"], i))[
                            0
                        ].squeeze(0),
                    )
                    for i in boards
                    if all([elem[1] not in excluded_positions for elem in i])
                ],
                batch_size=20 * (9-nes),
            )
            for tf in temp_transforms.keys()
        }

        num_epochs = 5000
        losses = np.zeros((num_epochs,))
        for tf, tf_training_data in temp_tf_training_data.items():

            t_losses, _, _, t_tf = train_transform(
                tf_training_data,
                num_epochs,
                simple_encoder,
                None,
                simple_decoder,
                None,
                temp_transforms[tf],
                temp_tf_optims[tf],
                loss_fn=ohe_loss_fn,
                noise_std=2.0,
                tf_name=None,
                logger=None,
            )

            losses += t_losses
            temp_transforms[tf] = t_tf


        for l in losses:
            if args.wandb: run.log({f"loss/{nes}_unseen_position_tf_mean": l})

        with torch.no_grad():
            results = search_test(
                list(itertools.chain(*shifts_eval_examples)),
                simple_encoder,
                simple_decoder,
                one_hot_tensor_represent,
                one_hot_tensor_represent,
                temp_transforms,
                apply_nn_transform,
                pruned_search_creator(single_object_ohe_hit_check, 7, args.timeout),
            )

        print(f"\n{nes} Unseen Positions: ", results["summary"])
        unseen_position_results["hit_rate"][nes] = results["summary"]["hit_rate"]
        unseen_position_results["timeout_rate"][nes] = results["summary"]["timeout_rate"]
        unseen_position_results["max_depth_rate"][nes] = results["summary"]["max_depth_rate"]
        if log_path is not None:
            pickle.dump(
                results,
                open(log_path.joinpath(f"{nes}_unseen_positions_results.ckpt"), "wb"),
            )


    fig = plt.figure()
    for k, v in unseen_position_results.items(): plt.plot(v, label=k)
    plt.title("Generalisation to Unseen Positions")
    plt.ylabel("Ratio")
    plt.xlabel("Number of Unseen Positions")
    plt.xticks(np.arange(0, 9, 1.0))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid()
    plt.legend()
    if log_path is not None: plt.savefig(log_path.joinpath("unseen_position_accuracy.png"))
    if args.wandb: run.log({"unseen_position_accuracy": fig})



if args.search or args.run_all:
    t_prog = ["out", "shiftright", "to-circle", "out",  "to-triangle", "shiftright", "out", "shiftdown", "to-square", "out", "shiftleft", "out"]
    t_board = {("triangle", (0, 0))}
    t_ex = generate_examples_exhaustive([t_prog], [t_board], lib)

    dummy_hit_check = lambda x, y: False
    unpruned_exhaustive_search_timings = []
    print("Evaluating Unpruned Search:")
    print(f"Depth\tFound\tTime")
    for i in range(1, 6):

        with torch.no_grad():
            t_search = exhaustive_search_creator(ohe_hit_check, i)
            r, s, time_taken = t_search(
                t_ex,
                one_hot_tensor_represent,
                one_hot_tensor_represent,
                simple_encoder,
                simple_decoder,
                simple_transforms,
                apply_nn_transform,
            )
            unpruned_exhaustive_search_timings.append(time_taken)
            print(f"{i}\t{r == 'success'}\t{unpruned_exhaustive_search_timings[-1]}")

    t_examples = []
    for i in range(20):
        t_examples.append(
            generate_examples_random(programs_upto_20[i], boards, lib, 100)
        )
    exhaustive_search_timings = collections.defaultdict(lambda : [])
    
    print("Evaluating Pruned Search -")
    for t_ex in tqdm(list(itertools.chain(*t_examples))):
        with torch.no_grad():
            t_depth = len(t_ex["program"]) + 1
            t_search = pruned_search_creator(single_object_ohe_hit_check, t_depth)
            r, s, time_taken = t_search(
                [t_ex],
                one_hot_tensor_represent,
                one_hot_tensor_represent,
                simple_encoder,
                simple_decoder,
                simple_transforms,
                apply_nn_transform,
            )
            t_depth = t_depth if r != "success" else len(s[0])
            exhaustive_search_timings[t_depth].append(time_taken)
        
    x, y = zip(*[(k, np.mean(v)) for k, v in exhaustive_search_timings.items()])
    fig = plt.figure()
    plt.plot(x, y, label='Pruned Emperical')
    plt.plot(range(1, len(unpruned_exhaustive_search_timings)+1), unpruned_exhaustive_search_timings, label='Unpruned')
    plt.title("Performance of Exhaustive Search")
    plt.ylabel("Time taken")
    plt.xlabel("Search Depth")
    plt.xticks(np.arange(1, max(x), 1.0))
    plt.grid()
    plt.legend()
    if log_path is not None: plt.savefig(log_path.joinpath("exhaustive_search_performance.png"))
    if args.wandb: run.log({"exhaustive_search_performance": fig})

if args.wandb: run.finish()
