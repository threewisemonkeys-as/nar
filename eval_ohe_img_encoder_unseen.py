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
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.datagen import (
    generate_board_states,
    generate_examples_random,
)
from src.models import apply_nn_transform, create_mlp, ConvEncoder
from src.ohe import ohe_fns_creator
from src.search import exhaustive_search_creator, pruned_search_creator, search_test
from src.training import train_transform, target_training, reconstruction_training
from src.image import img_represent_fns_creator, load_shape_map, IMG_SIZE


parser = argparse.ArgumentParser(description='Script to evaluate system with latent space based on one-hot represetation on unseen shapes')
parser.add_argument("--expID", type=int, choices=[1, 2, 3, 4], help="Which experiment to perform. Check README for details")
parser.add_argument("--transform_training_epochs", type=int, default=None, help="Perform transform training for this many epochs. Default behaviour is to load trained from data/")
parser.add_argument("--save_transforms", action="store_true", help="Whether to save transforms. (--transform_training_epochs must be specified)")
parser.add_argument("--reconstruction_training_epochs", type=int, default=None, help="Perform reconstruction training to train one-hot based encoder decoder for this many epochs. Default behaviour is to load trained from data/")
parser.add_argument("--save_latent", action="store_true", help="Whether to save encoder and decoder. (--reconstruction_training_epochs must be specified)")
parser.add_argument("--img_encoder_training_epochs", type=int, default=None, help="Perform image encoder target training with this many epcohcs")
parser.add_argument("--save_img_encoder", action="store_true", help="Whether to save image encoder. (--img_encoder_training_epochs must be specified)")
parser.add_argument("--eval_n", type=int, default=0, help="Evaluate system on search with this many examples per level upto 20")
parser.add_argument("--wandb", action="store_true", help="Log to wandb")
parser.add_argument("--log_path", type=str, default="results/ohe_img_encoder_unseen", help="Path to store logs")
parser.add_argument("--data_path", type=str, default="data/", help="Path where library, images etc.. are stored")
parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
parser.add_argument("--num_cpu_training", type=int, default=8, help="Max number of cpu to use for parallelism for training.")
parser.add_argument("--num_cpu_search", type=int, default=16, help="Max number of cpu to use for parallelism for search")
parser.add_argument("--timeout", type=int, default=60, help="Timeout for search in seconds")
parser.add_argument("--num_symbolic_shapes", type=int, default=4, choices=range(0, 21), help="Number of shapes to train symbolic encoder/decoder and transforms on. (Only valid for experiment 3 and 4)")
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
shift_programs_upto_20 = pickle.load(open(data_path.joinpath("programs/shift_programs_upto_20.pkl"), "rb"))

shapes = ["circle", "square", "triangle", "delta", "b", "d", "e", "g", "k", "m", "r", "s", "u", "w", "x", "z", "theta", "pi", "tau", "psi"]

if args.expID in [1, 2]:
    t_num_symbolic_shapes = 0
    exp_lib = shift_lib
    exp_progs = shift_programs_upto_20

if args.expID in [3, 4]:
    t_num_symbolic_shapes = args.num_symbolic_shapes
    exp_lib = lib
    exp_progs = programs_upto_20


shapes_for_symbolic = ["unseen"] + shapes[:t_num_symbolic_shapes]
boards_for_symbolic = generate_board_states(shapes_for_symbolic, 1)

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
) = ohe_fns_creator(shapes_for_symbolic, 3)

ohe_loss_fn = ohe_loss_fn_creator()
one_hot_tensor_represent = one_hot_tensor_represent_creator(device, dtype)
one_hot_tensor_represent_cpu = one_hot_tensor_represent_creator(torch.device("cpu"), dtype)

one_hot_tensor_with_unseen_represent = lambda b: one_hot_tensor_represent({("unseen", pos) if s not in shapes_for_symbolic else (s, pos) for (s, pos) in b})
one_hot_tensor_with_unseen_represent_cpu = lambda b: one_hot_tensor_represent_cpu({("unseen", pos) if s not in shapes_for_symbolic else (s, pos) for (s, pos) in b})

latent_dim = 32
input_dim = sum(data_split)
img_encoder_lr = 3e-4
tf_lr = 3e-4    
encoder_lr = 3e-4
decoder_lr = 3e-4

shape_map = load_shape_map("data/images")

single_img_represent, single_img_tensor_represent = img_represent_fns_creator(
    shape_map, device, dtype
)
_, single_img_tensor_represent_cpu = img_represent_fns_creator(
    shape_map, torch.device("cpu"), dtype
)


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
            [one_hot_tensor_represent(b)[0].squeeze(0) for b in boards_for_symbolic],
            batch_size=len(boards_for_symbolic),
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
            pickle.dump(copy.deepcopy(simple_encoder).to(torch.device("cpu")), open(data_path.joinpath(f"weights/ohe_img_encoder_unseen/simple_encoder_exp{args.expID}.pkl"), "wb"))
            pickle.dump(copy.deepcopy(simple_decoder).to(torch.device("cpu")), open(data_path.joinpath(f"weights/ohe_img_encoder_unseen/simple_decoder_exp{args.expID}.pkl"), "wb"))

    else:

        # load enoder and decoder
        simple_encoder = pickle.load(open(data_path.joinpath(f"weights/ohe_img_encoder_unseen/simple_encoder_exp{args.expID}.pkl"), "rb"))
        simple_decoder = pickle.load(open(data_path.joinpath(f"weights/ohe_img_encoder_unseen/simple_decoder_exp{args.expID}.pkl"), "rb"))

    # get cpu versions of encoder and decoder
    simple_encoder_cpu = copy.deepcopy(simple_encoder).to(torch.device("cpu"))
    simple_decoder_cpu = copy.deepcopy(simple_decoder).to(torch.device("cpu"))

    ##########################
    ### Transform training ###
    ##########################

    if args.transform_training_epochs is not None:

        simple_transforms = {
            k: create_mlp(latent_dim, latent_dim, [32]).to(dtype).to(device)
            for k in exp_lib.primitives_dict.keys()
        }
        simple_tf_optims = {
            k: torch.optim.Adam(tf_net.parameters(), lr=tf_lr)
            for k, tf_net in simple_transforms.items()
        }

        all_tf_training_data = {
            tf: torch.utils.data.DataLoader(
                [
                    (
                        one_hot_tensor_with_unseen_represent(i)[0].squeeze(0),
                        one_hot_tensor_with_unseen_represent(exp_lib.apply_program([tf, "out"], i))[0].squeeze(
                            0
                        ),
                    )
                    for i in boards_for_symbolic
                ],
                batch_size=len(boards_for_symbolic),
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

        # save nets using pickle
        if args.save_transforms:
            simple_transforms_cpu = {k: copy.deepcopy(v).to(torch.device("cpu")) for k, v in simple_transforms.items()}
            pickle.dump(simple_transforms_cpu, open(data_path.joinpath(f"weights/ohe_img_encoder_unseen/simple_transforms_exp{args.expID}.pkl"), "wb"))

    else:
        # load transforms
        simple_transforms = pickle.load(open(data_path.joinpath(f"weights/ohe_img_encoder_unseen/simple_transforms_exp{args.expID}.pkl"), "rb"))

    simple_transforms_cpu = {k: copy.deepcopy(v).to(torch.device("cpu")) for k, v in simple_transforms.items()}

    ##############################
    ### Image encoder training ###
    ##############################

    if args.img_encoder_training_epochs is not None:

        pool = mp.Pool(min(20, args.num_cpu_training))
        jobs = {}    
        img_encoders = {}

        for t_num_img_encoder_shapes in range(20, 0, -1):
            shapes_for_img_encoder = shapes[:t_num_img_encoder_shapes]
            boards_for_img_encoder = generate_board_states(shapes_for_img_encoder, 1)
            
            # instantiate nets
            img_encoder = ConvEncoder(IMG_SIZE, latent_dim, True).to(device).to(dtype)
            img_encoder_optim = torch.optim.Adam(img_encoder.parameters(), lr=img_encoder_lr)

            # training 
            xs = torch.utils.data.DataLoader(
                [single_img_tensor_represent(b)[0].squeeze(0) for b in boards_for_img_encoder],
                batch_size=len(boards_for_img_encoder),
            )
            with torch.no_grad():
                ys = torch.utils.data.DataLoader(
                    [
                        simple_encoder(one_hot_tensor_with_unseen_represent(b)[0].squeeze(0))
                        for b in boards_for_img_encoder
                    ],
                    batch_size=len(boards_for_img_encoder),
                )

            r = pool.apply_async(
                target_training,
                (
                    xs,
                    ys,
                    args.img_encoder_training_epochs,
                    img_encoder,
                    img_encoder_optim,
                    F.mse_loss,
                    0.0,
                ),
            )
            jobs[t_num_img_encoder_shapes] = r

        for t_num_img_encoder_shapes, j in jobs.items():
            losses, img_encoder = j.get()
            img_encoders[t_num_img_encoder_shapes] = img_encoder
            
        pool.close() 
        pool.terminate()
        pool.join()

        if args.save_img_encoder:
            # save net using pickle
            img_encoders_cpu= {k: copy.deepcopy(v).to(torch.device("cpu")) for k, v in img_encoders.items()}
            pickle.dump(img_encoders_cpu, open(data_path.joinpath(f"weights/ohe_img_encoder_unseen/img_encoders_exp{args.expID}.pkl"), "wb"))

    else:
        # load img_encoders
        img_encoders = pickle.load(open(data_path.joinpath(f"weights/ohe_img_encoder_unseen/img_encoders_exp{args.expID}.pkl"), "rb"))

    img_encoders_cpu= {k: copy.deepcopy(v).to(torch.device("cpu")) for k, v in img_encoders.items()}

    ##########################
    ### Transform training ###
    ##########################

    if args.eval_n > 0:

        if args.expID in [1, 3]:
            test_boards = generate_board_states(shapes[:20], 1)
            eval_examples = []
            for i in range(20):
                eval_examples.append(
                    generate_examples_random(exp_progs[i], test_boards, exp_lib, args.eval_n)
                )
            eval_examples = list(itertools.chain(*eval_examples))

            pbar = tqdm(total=len(img_encoders_cpu) * len(eval_examples))

        elif args.expID in [2, 4]:
            t_n = 0
            eval_examples_dict = {}
            for t_num_img_encoder_shapes in img_encoders_cpu.keys():
                if t_num_img_encoder_shapes == 20: continue
                test_shapes = shapes[t_num_img_encoder_shapes:]
                test_boards = generate_board_states(test_shapes, 1)
                t_eval_examples = []
                for i in range(20):
                    t_eval_examples.append(
                        generate_examples_random(exp_progs[i], test_boards, exp_lib, args.eval_n)
                    )
                t_eval_examples = list(itertools.chain(*t_eval_examples))
                eval_examples_dict[t_num_img_encoder_shapes] = t_eval_examples
                t_n += len(t_eval_examples)

            pbar = tqdm(total=t_n)
        
        pbar_update = lambda *a: pbar.update()
        pool = mp.Pool(args.num_cpu_search)
        jobs = {k: [] for k in img_encoders_cpu.keys()}
        t_search = pruned_search_creator(single_object_ohe_hit_check, 21, args.timeout)

        for t_num_img_encoder_shapes, img_encoder_cpu in img_encoders_cpu.items():
            if args.expID in [2, 4]:
                if t_num_img_encoder_shapes == 20: continue
                eval_examples = eval_examples_dict[t_num_img_encoder_shapes]

            for t_ex in eval_examples:
                jobs[t_num_img_encoder_shapes].append(
                    pool.apply_async(
                        t_search,
                        args=(
                            [t_ex],
                            single_img_tensor_represent_cpu,
                            one_hot_tensor_with_unseen_represent_cpu,
                            img_encoder_cpu,
                            simple_decoder_cpu,
                            simple_transforms_cpu,
                            apply_nn_transform,
                        ),
                        callback=pbar_update,
                    )
                )
        
        results = {k: {kk: 0 for kk in ["n", "hits", "timeouts", "max_depths"]} for k in img_encoders.keys()}
        for t_num_img_encoder_shapes, t_jobs in jobs.items():
            for j in t_jobs:
                result, solution, time_taken = j.get()
                results[t_num_img_encoder_shapes]["n"] += 1
                if result == "success":
                    results[t_num_img_encoder_shapes]["hits"] += 1
                elif result == "timeout":
                    results[t_num_img_encoder_shapes]["timeouts"] += 1
                elif result == "max_depth":
                    results[t_num_img_encoder_shapes]["max_depths"] += 1

        pool.close() 
        pool.terminate()
        pool.join()

        hits = {}
        timeouts = {}
        max_depths = {}

        print("Shapes Seen\tHits\tTimeouts\tMax Depths")
        for t_num_img_encoder_shapes, r in results.items():
            if r["n"] > 0:
                print(f"{t_num_img_encoder_shapes}\t{r['hits'] / r['n']:.2f}\t{r['timeouts'] / r['n']:.2f}\t{r['max_depths'] / r['n']:.2f}")
                hits[t_num_img_encoder_shapes] = r["hits"] / r["n"]
                timeouts[t_num_img_encoder_shapes] = r["timeouts"] / r["n"]
                max_depths[t_num_img_encoder_shapes] = r["max_depths"] / r["n"]
            
        fig = plt.figure()
        plt.plot(hits.keys(), hits.values(), label="hits")
        plt.plot(timeouts.keys(), timeouts.values(), label="timeouts")
        plt.plot(max_depths.keys(), max_depths.values(), label="max_depths")
        plt.title(f"Generalisation of Encoder Unseen Shapes (Exp {args.expID})")
        plt.ylabel("Ratio")
        plt.xlabel("Number of Shapes Seen by Encoder")
        plt.xticks(np.arange(0, 21, 1.0))
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.grid()
        plt.legend()
        if log_path is not None: plt.savefig(log_path.joinpath(f"ohe_img_encoder_unseen_exp{args.expID}_accuracy.png"))
        if args.wandb: run.log({f"ohe_img_encoder_unseen_exp{args.expID}_accuracy": fig})
