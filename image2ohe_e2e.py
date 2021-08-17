import argparse
import pathlib
import pickle

import yaml
import torch
import numpy as np
import wandb

from src.models import apply_nn_transform
from src.ohe import ohe_fns_creator
from src.search import pruned_search_raw_creator
from src.image import  homography_image_match_creator, load_img, split_image
from src.utils import apply_transform_program_raw, visualise_example_with_unseen


parser = argparse.ArgumentParser(description='Script to solve problem')
parser.add_argument("--task_dir", type=str, default=None, help="Path of IO specification")
parser.add_argument("--encoder", type=str, default=None, help="Path of the encoder model")
parser.add_argument("--decoder", type=str, default=None, help="Path of the decoder model")
parser.add_argument("--transforms", type=str, default=None, help="Path of the transform model")
parser.add_argument("--data_path", type=str, default="data/", help="Path where library, images etc.. are stored")
parser.add_argument("--log_path", type=str, default="results/", help="Path to store logs")
parser.add_argument("--output_path", type=str, default="e2e_output.png", help="Path to store output")
parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
parser.add_argument("--timeout", type=int, default=60, help="Timeout for search in seconds")
parser.add_argument("--max_depth", type=int, default=21, help="Timeout for search in seconds")
parser.add_argument("--wandb", action="store_true", help="Log to wandb")
parser.add_argument("--shape_slots", type=int, default=1, help="Number of shapes slots present in one-hot (including unseen)")

args = parser.parse_args()


##################################
### Set up logging and devices ###
##################################

run = wandb.init(project="nar", entity="atharv", group="ohe_img_encoder_unseen") if args.wandb else None
log_path = pathlib.Path(args.log_path) if args.log_path is not None else pathlib.Path(run.dir) if args.wandb else None
data_path = pathlib.Path(args.data_path)

if log_path is not None: log_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu") 
dtype = torch.float

################################
### Load models and function ###
################################

encoder = pickle.load(open(args.encoder, "rb"))
decoder = pickle.load(open(args.decoder, "rb"))
transforms = pickle.load(open(args.transforms, "rb"))

encoder.to(device).to(dtype)
decoder.to(device).to(dtype)
for tf in transforms.values(): tf.to(device).to(dtype)

encoder.eval()
decoder.eval()
for tf in transforms.values(): tf.eval()

num_shapes = args.shape_slots

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
) = ohe_fns_creator(list(range(num_shapes)), 3)

search_fn = pruned_search_raw_creator(single_object_ohe_hit_check, args.max_depth, args.timeout)


##########################
### Load the task data ###
##########################

task_dir = pathlib.Path(args.task_dir)
if not task_dir.exists(): raise ValueError(f"Task directory {task_dir} does not exist")
task_spec = yaml.load(open(task_dir / "spec.yml", "r"), yaml.SafeLoader)
task = {}
if task_spec["type"] == "image":
    task = {
        "examples": [
            {"input": load_img(task_dir / e["input"]), 
            "output": load_img(task_dir / e["output"])}
            for e in task_spec["examples"]
        ],
        "query": load_img(task_dir / task_spec["query"]),
    }
else:
    raise NotImplementedError(f"Type: {task_spec['type']} is not implemented")


###########################
### Preprocess the data ###
###########################

h_img_match = homography_image_match_creator(0.5)

for tidx, e in enumerate(task["examples"]):
    o_tags = []
    t_i_imgs = split_image(e["input"])
    t_o_imgs = split_image(e["output"])
    for t_o_img in t_o_imgs:
        t_match = False
        for i_idx, t_i_img in enumerate(t_i_imgs):
            if(h_img_match(t_o_img, t_i_img)):
                t_match = True 
                o_tags.append(i_idx)
                break
            if not t_match:
                raise ValueError("Shape found in output not matching any shape in input")

    task["examples"][tidx] = {
        "input": [
            (idx, torch.tensor(np.asarray(elem), device=device, dtype=dtype).unsqueeze(0) / 255) 
            for idx, elem in enumerate(t_i_imgs)
        ],
        "output": [
            (idx, decoder(encoder(torch.tensor(np.asarray(elem), device=device, dtype=dtype).unsqueeze(0) / 255)))
            for idx, elem in zip(o_tags, t_o_imgs)
        ],
    }

######################
### Perform Search ###
######################

print("Beginning search ...", flush=True, end=' ')
r = search_fn(
    task["examples"],
    encoder,
    decoder,
    transforms,
    apply_nn_transform,
)
print("Completed!")

print(f"\tResult: {r[0]}\n\tFound: {r[1][0]}\n\tTime taken: {r[2]:.5f} s\n")

######################
### Apply to query ###
######################

i = [
    torch.tensor(np.asarray(elem), device=device, dtype=dtype).unsqueeze(0) / 255
    for elem in split_image(task["query"])
]

f = apply_transform_program_raw(
    r[1][0],
    i,
    encoder,
    decoder,
    transforms,
    apply_nn_transform,
)

out = visualise_example_with_unseen(list(enumerate(split_image(task["query"]))), f, ohe_decode)
out.save(args.output_path)
