import argparse
import copy
import itertools
import pathlib
import pickle
import random

import dill
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import torch
import wandb

from src.datagen import (
    generate_board_states,
    generate_examples_random,
)
from src.models import apply_nn_transform, create_mlp, ConvEncoder
from src.ohe import ohe_fns_creator
from src.search import pruned_search_creator, search_test
from src.training import train_transform


parser = argparse.ArgumentParser(description='Script to evaluate system with latent space based on one-hot represetation on shapes and positions unseen by transforms')
parser.add_argument("--unseen_shapes", action="store_true", help="Run shapes unseen by transforms experiment")
parser.add_argument("--unseen_positions", action="store_true", help="Run positions unseen by transforms experiment")
parser.add_argument("--wandb", action="store_true", help="Log to wandb")
parser.add_argument("--log_path", type=str, default="results/ohe_transforms_unseen", help="Path to store logs")
parser.add_argument("--data_path", type=str, default="data/", help="Path where library, images etc.. are stored")
parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
parser.add_argument("--num_cpu_training", type=int, default=8, help="Max number of cpu to use for parallelism for training.")
parser.add_argument("--num_cpu_search", type=int, default=16, help="Max number of cpu to use for parallelism for search")
parser.add_argument("--timeout", type=int, default=60, help="Timeout for search in seconds")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu to use")
parser.add_argument("--disable_gpu", action="store_true", help="Flag to disable gpu even if it is available. Default behaviour is to use GPU if available")
args = parser.parse_args()


#########################################
### Set up seeds, logging and devices ###
#########################################

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if mp.cpu_count() < max(args.num_cpu_training, args.num_cpu_search):
    print(f"WARNING: Not enough cpu ({mp.cpu_count()}) for desired parallelism {max(args.num_cpu_training, args.num_cpu_search)}")

run = wandb.init(project="nar", entity="atharv", group="ohe_transforms_unseen") if args.wandb else None
log_path = pathlib.Path(args.log_path) if args.log_path is not None else pathlib.Path(run.dir) if args.wandb else None
data_path = pathlib.Path(args.data_path)

if log_path is not None: log_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu" if (not torch.cuda.is_available()) or args.disable_gpu else f"cuda:{args.gpu_id}")
dtype = torch.float

#################################
### Set up the model and data ###
#################################

lib = dill.load(open(data_path.joinpath("libraries/shift_library0.pkl"), "rb"))
shift_lib = dill.load(open(data_path.joinpath("libraries/shift_library0.pkl"), "rb"))
programs_upto_20 = pickle.load(open(data_path.joinpath("programs/programs_upto_20.pkl"), "rb"))
shift_programs_upto_6 = pickle.load(open(data_path.joinpath("programs/shift_programs_upto_6.pkl"), "rb"))

shapes = ["circle", "square", "triangle", "delta", "b", "d", "e", "g", "k", "m", "r", "s", "u", "w", "x", "z", "theta", "pi", "tau", "psi"]
test_boards = generate_board_states(shapes, 1)

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
one_hot_tensor_represent_cpu = one_hot_tensor_represent_creator(torch.device("cpu"), dtype)

latent_dim = 32
input_dim = sum(data_split)
tf_lr = 3e-4    


if __name__=='__main__':

    mp.set_start_method("spawn")

    ################################
    ### Load encoder and decoder ###
    ################################

    simple_encoder = pickle.load(open(data_path.joinpath("weights/simple_encoder.pkl"), "rb")).to(device)
    simple_decoder = pickle.load(open(data_path.joinpath("weights/simple_decoder.pkl"), "rb")).to(device)

    simple_encoder_cpu = copy.deepcopy(simple_encoder).to(torch.device("cpu"))
    simple_decoder_cpu = copy.deepcopy(simple_decoder).to(torch.device("cpu"))


    ##############################
    ### Generate Test Examples ###
    ##############################

    shifts_eval_examples = []
    for i in range(6):
        shifts_eval_examples.append(
            generate_examples_random(
                shift_programs_upto_6[i],
                test_boards,
                lib,
                50,
            )
        )

    #####################
    ### Unseen Shapes ###
    #####################

    if args.unseen_shapes:

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
                        for i in test_boards
                        if all([elem[0] not in excluded_shapes for elem in i])
                    ],
                    batch_size=(20 - nes) * 9,
                )
                for tf in temp_transforms.keys()
            }

            t_results = {}
            num_epochs = 5000
            losses = np.zeros((num_epochs,))
            pool = mp.Pool(min(len(temp_transforms), args.num_cpu_search))
            for tf, tf_training_data in temp_tf_training_data.items():
                r = pool.apply_async(
                    train_transform,
                    (
                        tf_training_data,
                        num_epochs,
                        simple_encoder,
                        None,
                        simple_decoder,
                        None,
                        temp_transforms[tf],
                        temp_tf_optims[tf],
                        ohe_loss_fn,
                        2.0,
                    )
                )
                t_results[tf] = r
            
            for tf, r in t_results.items():
                t_l, _, _, t_tf = r.get()
                losses += t_l
                temp_transforms[tf] = copy.deepcopy(t_tf)
                del t_tf

            pool.close()
            pool.terminate()
            pool.join()

            for t_tf in temp_transforms.values(): t_tf.to(torch.device("cpu"))

            with torch.no_grad():
                results = search_test(
                    list(itertools.chain(*shifts_eval_examples)),
                    simple_encoder_cpu,
                    simple_decoder_cpu,
                    one_hot_tensor_represent_cpu,
                    one_hot_tensor_represent_cpu,
                    temp_transforms,
                    apply_nn_transform,
                    pruned_search_creator(single_object_ohe_hit_check, 7, args.timeout),
                    n_workers=args.num_cpu_search,
                )

            print(f"\n{nes} Unseen Shapes: ", results["summary"])
            unseen_shape_results["hit_rate"][nes] = results["summary"]["hit_rate"]
            unseen_shape_results["timeout_rate"][nes] = results["summary"]["timeout_rate"]
            unseen_shape_results["max_depth_rate"][nes] = results["summary"]["max_depth_rate"]


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


    #####################
    ### Unseen Shapes ###
    #####################

    if args.unseen_positions:

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
                        for i in test_boards
                        if all([elem[1] not in excluded_positions for elem in i])
                    ],
                    batch_size=20 * (9-nes),
                )
                for tf in temp_transforms.keys()
            }

            num_epochs = 5000
            t_results = {}
            num_epochs = 5000
            losses = np.zeros((num_epochs,))
            pool = mp.Pool(min(len(temp_transforms), args.num_cpu_training))
            for tf, tf_training_data in temp_tf_training_data.items():
                r = pool.apply_async(
                    train_transform,
                    (
                        tf_training_data,
                        num_epochs,
                        simple_encoder,
                        None,
                        simple_decoder,
                        None,
                        temp_transforms[tf],
                        temp_tf_optims[tf],
                        ohe_loss_fn,
                        2.0,
                    )
                )
                t_results[tf] = r
            
            for tf, r in t_results.items():
                t_l, _, _, t_tf = r.get()
                losses += t_l
                temp_transforms[tf] = copy.deepcopy(t_tf)
                del t_tf

            pool.close() 
            pool.terminate()
            pool.join()

            for t_tf in temp_transforms.values(): t_tf.to(torch.device("cpu"))

            with torch.no_grad():
                results = search_test(
                    list(itertools.chain(*shifts_eval_examples)),
                    simple_encoder_cpu,
                    simple_decoder_cpu,
                    one_hot_tensor_represent_cpu,
                    one_hot_tensor_represent_cpu,
                    temp_transforms,
                    apply_nn_transform,
                    pruned_search_creator(single_object_ohe_hit_check, 7, args.timeout),
                    n_workers=args.num_cpu_search,
                )

            print(f"\n{nes} Unseen Positions: ", results["summary"])
            unseen_position_results["hit_rate"][nes] = results["summary"]["hit_rate"]
            unseen_position_results["timeout_rate"][nes] = results["summary"]["timeout_rate"]
            unseen_position_results["max_depth_rate"][nes] = results["summary"]["max_depth_rate"]

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

    if args.wandb: run.finish()
