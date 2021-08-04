import argparse
import collections
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
parser.add_argument("--transform_training_epochs", type=int, default=None, help="Perform transform training for this many epochs. Default behaviour is to load trained from data/")
parser.add_argument("--save_transforms", action="store_true", help="Whether to save transforms. (--log_path and --transform_training_epochs must be specified)")
parser.add_argument("--reconstruction_training_epochs", type=int, default=None, help="Perform reconstruction training to train one-hot based encoder decoder for this many epochs. Default behaviour is to load trained from data/")
parser.add_argument("--save_latent", action="store_true", help="Whether to save encoder and decoder. (--log_path and --reconstruction_training_epochs must be specified)")
parser.add_argument("--eval_n", type=int, default=None, help="Evaluate system on search with this many examples per level upto 20")
parser.add_argument("--eval_latent_acc_n", type=int, default=None, help="Evaluate accuracy of transforms with thismany examples per level upto 20")
parser.add_argument("--search_n", type=int, default=None, help="Evaluate search performance with thismany examples per level upto 20")
parser.add_argument("--wandb", action="store_true", help="Log to wandb")
parser.add_argument("--log_path", type=str, default=None, help="Path to store logs")
parser.add_argument("--data_path", type=str, default="data/", help="Path where library, model weights etc.. are stored")
parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
parser.add_argument("--timeout", type=int, default=120, help="Timeout for search in seconds")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu to use")
parser.add_argument("--num_cpu_training", type=int, default=8, help="Max number of cpu to use for parallelism for training.")
parser.add_argument("--num_cpu_search", type=int, default=16, help="Max number of cpu to use for parallelism for search")
parser.add_argument("--disable_gpu", action="store_true", help="Flag to disable gpu even if it is available")
args = parser.parse_args()

#########################################
### Set up seeds, logging and devices ###
#########################################

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if mp.cpu_count() < max(args.num_cpu_training, args.num_cpu_search):
    print(f"WARNING: Not enough cpu ({mp.cpu_count()}) for desired parallelism {max(args.num_cpu_training, args.num_cpu_search)}")


run = wandb.init(project="nar", entity="atharv", group="ohe") if args.wandb else None
log_path = pathlib.Path(args.log_path) if args.log_path is not None else pathlib.Path(run.dir) if args.wandb else None
data_path = pathlib.Path(args.data_path)

if log_path is not None: log_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu" if (not torch.cuda.is_available()) or args.disable_gpu else f"cuda:{args.gpu_id}")
dtype = torch.float

#################################
### Set up the model and data ###
#################################

lib = dill.load(open(data_path.joinpath("libraries/library0.pkl"), "rb"))
shift_lib = dill.load(open(data_path.joinpath("libraries/shift_library0.pkl"), "rb"))
programs_upto_20 = pickle.load(open(data_path.joinpath("programs/programs_upto_20.pkl"), "rb"))
shift_programs_upto_6 = pickle.load(open(data_path.joinpath("programs/shift_programs_upto_6.pkl"), "rb"))

shapes = ["circle", "square", "triangle", "delta", "b", "d", "e", "g", "k", "m", "r", "s", "u", "w", "x", "z", "theta", "pi", "tau", "psi"]
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
one_hot_tensor_represent_cpu = one_hot_tensor_represent_creator(torch.device("cpu"), dtype)

input_dim = sum(data_split)
latent_dim = 32

# learning rates
encoder_lr = 3e-4
decoder_lr = 3e-4
tf_lr = 3e-4


if __name__=='__main__':

    mp.set_start_method("spawn")

    ################################################
    ### Reconstruction training for latent space ###
    ################################################

    if args.reconstruction_training_epochs is not None:

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
            args.reconstruction_training_epochs,
            simple_encoder,
            simple_encoder_optim,
            simple_decoder,
            simple_decoder_optim,
            loss_fn=ohe_loss_fn,
            noise_std=2.0,
            logger=run,
        )

        # save nets with pickle
        if args.save_transforms:
            pickle.dump(copy.deepcopy(simple_encoder).to(torch.device("cpu")), open(data_path.joinpath(f"weights/simple_encoder.pkl"), "wb"))
            pickle.dump(copy.deepcopy(simple_decoder).to(torch.device("cpu")), open(data_path.joinpath(f"weights/simple_decoder.pkl"), "wb"))

    else:

        simple_encoder = pickle.load(open(data_path.joinpath("weights/simple_encoder.pkl"), "rb")).to(device)
        simple_decoder = pickle.load(open(data_path.joinpath("weights/simple_decoder.pkl"), "rb")).to(device)

    simple_encoder_cpu = copy.deepcopy(simple_encoder).to(torch.device("cpu"))
    simple_decoder_cpu = copy.deepcopy(simple_decoder).to(torch.device("cpu"))

    ##########################
    ### Transform training ###
    ##########################

    if args.transform_training_epochs is not None:

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

        num_epochs = args.transform_training_epochs
        losses = np.zeros((num_epochs,))
        t_results = {}

        pool = mp.Pool(min(len(simple_transforms), args.num_cpu_training))
        for tf, tf_training_data in all_tf_training_data.items():
            r = pool.apply_async(
                train_transform,
                (
                    tf_training_data,
                    num_epochs,
                    simple_encoder,
                    None,
                    simple_decoder,
                    None,
                    simple_transforms[tf],
                    simple_tf_optims[tf],
                    ohe_loss_fn,
                    2.0,
                )
            )
            t_results[tf] = r
        
        for tf, r in t_results.items():
            t_l, _, _, t_tf = r.get()
            losses += t_l
            simple_transforms[tf] = copy.deepcopy(t_tf)
            del t_tf

        pool.close() 
        pool.terminate()
        pool.join()

        if args.wandb:
            for i, l in enumerate(losses):
                run.log({"loss/tf_mean": l, "epoch": i})

        if args.save_transforms:
            simple_transforms_cpu = {k: copy.deepcopy(v).to(torch.device("cpu")) for k, v in simple_transforms.items()}
            pickle.dump(simple_transforms_cpu, open(data_path.joinpath("weights/simple_transforms.pkl"), "wb"))

    else:
        simple_transforms = pickle.load(open(data_path.joinpath("weights/simple_transforms.pkl"), "rb"))
        for t_tf in simple_transforms.values(): t_tf.to(device)

    simple_transforms_cpu = {k: copy.deepcopy(v).to(torch.device("cpu")) for k, v in simple_transforms.items()}


    ##############################################
    ### Evaluation of accuracy of latent space ###
    ##############################################


    if args.eval_latent_acc_n is not None:
        tf_names = list(lib.primitives_dict.keys())
        t_n_progs = 500
        eval_examples = []
        for i in range(1, 20):
            t_progs = [random.choices(tf_names, k=i) + ["out"] for _ in range(t_n_progs)]
            eval_examples.append(
                generate_examples_random(t_progs, boards, lib, args.eval_n)
            )

        with torch.no_grad():
            results = search_test(
                list(itertools.chain(*eval_examples)),
                simple_encoder_cpu,
                simple_decoder_cpu,
                one_hot_tensor_represent_cpu,
                one_hot_tensor_represent_cpu,
                simple_transforms_cpu,
                apply_nn_transform,
                pruned_search_creator(single_object_ohe_hit_check, 21, args.timeout),
                n_workers=args.num_cpu_search,
            )
        
        n_dict = collections.defaultdict(lambda : 0)
        r_dict = {k: collections.defaultdict(lambda : 0) for k in ["hits", "timeout", "max_depth"]}
        for r in results["details"]:
            if r["result"] == "success":
                r_dict["hits"][len(r["solution"][0])] += 1
                n_dict[len(r["solution"][0])] += 1
            elif r["result"] == "timeout":
                r_dict["timeout"][len(r["example"]["program"])] += 1
                n_dict[len(r["example"]["program"])] += 1
            elif r["result"] == "max_depth":
                r_dict["max_depth"][len(r["example"]["program"])] += 1
                n_dict[len(r["example"]["program"])] += 1

        fig = plt.figure()
        for k, v in r_dict.items():
            t_dict = {}
            for ii, vv in v.items():
                t_dict[ii] = vv / n_dict[ii]
            plt.plot(list(t_dict.keys()), list(t_dict.values()), label=k)
        plt.title("Latent Space Accuracy")
        plt.ylabel("Ratio")
        plt.xlabel("Depth")
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.grid()
        plt.legend()
        if log_path is not None: plt.savefig(log_path.joinpath("ohe/latent_space_accuracy.png"))


    #################################################
    ### Evaluation through search of latent space ###
    #################################################


    if args.eval_n is not None:

        eval_examples = []
        for i in range(20):
            eval_examples.append(
                generate_examples_random(programs_upto_20[i], boards, lib, args.eval_n)
            )

        with torch.no_grad():
            results = search_test(
                list(itertools.chain(*eval_examples)),
                simple_encoder_cpu,
                simple_decoder_cpu,
                one_hot_tensor_represent_cpu,
                one_hot_tensor_represent_cpu,
                simple_transforms_cpu,
                apply_nn_transform,
                pruned_search_creator(single_object_ohe_hit_check, 21, args.timeout),
                n_workers=args.num_cpu_search,
            )

        print("\n", results["summary"])
        if args.wandb: run.summary.extend(results["summary"])
        if log_path is not None: pickle.dump(results, open(log_path.joinpath("ohe/eval_results.ckpt"), "wb"))


    ########################################
    ### Evaluation of search performance ###
    ########################################


    if args.search_n is not None:
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
                    one_hot_tensor_represent_cpu,
                    one_hot_tensor_represent_cpu,
                    simple_encoder_cpu,
                    simple_decoder_cpu,
                    simple_transforms_cpu,
                    apply_nn_transform,
                )
                unpruned_exhaustive_search_timings.append(time_taken)
                print(f"{i}\t{r == 'success'}\t{unpruned_exhaustive_search_timings[-1]}")

        t_examples = []
        for i in range(20):
            t_examples.append(
                generate_examples_random(programs_upto_20[i], boards, lib, args.search_n)
            )
        exhaustive_search_timings = collections.defaultdict(lambda : [])
        
        print("Evaluating Pruned Search -")
        t_data = list(itertools.chain(*t_examples))
        t_pbar = tqdm(total=len(t_data))
        t_pbar_update = lambda *a: t_pbar.update()
        pool = mp.Pool(processes=args.num_cpu_search)
        jobs = []
        for t_ex in t_data:
            with torch.no_grad():
                t_depth = len(t_ex["program"]) + 1
                t_search = pruned_search_creator(single_object_ohe_hit_check, t_depth)
                jobs.append(
                    pool.apply_async(
                        t_search,
                        args=(
                            [t_ex],
                            one_hot_tensor_represent_cpu,
                            one_hot_tensor_represent_cpu,
                            simple_encoder_cpu,
                            simple_decoder_cpu,
                            simple_transforms_cpu,
                            apply_nn_transform,
                        ),
                        callback=t_pbar_update
                    )
                )

        for j, t_ex in zip(jobs, t_data):
            t_depth = len(t_ex["program"]) + 1
            r, s, t = j.get()
            t_depth = t_depth if r != "success" else len(s[0])
            exhaustive_search_timings[t_depth].append(t)
        
        pool.close()
        pool.terminate()
        pool.join()

        x, y = zip(*[(k, np.mean(v)) for k, v in exhaustive_search_timings.items()])
        x, y = zip(*sorted(zip(x, y)))
        
        print("Depth\tFound\tTime")
        for i in range(len(x)):
            print(f"{x[i]}\t{y[i]}")
        

        fig = plt.figure()
        plt.plot(x, y, label='Pruned Emperical')
        plt.plot(range(1, len(unpruned_exhaustive_search_timings)+1), unpruned_exhaustive_search_timings, label='Unpruned')
        plt.title("Performance of Exhaustive Search")
        plt.ylabel("Time taken")
        plt.xlabel("Search Depth")
        plt.xticks(np.arange(1, max(x)+1, 1.0))
        plt.grid()
        plt.legend()
        if log_path is not None: plt.savefig(log_path.joinpath("ohe/exhaustive_search_performance.png"))
        if args.wandb: run.log({"exhaustive_search_performance": fig})

    if args.wandb: run.finish()
