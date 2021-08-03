# nar

Neural Algorithms for Analogical Reasoning

## Setup

1. Setup an environment with `python>=3.8`.
2. Install all the dependencies for the project: `pip install -r requirements.txt`

## Project Structure

### Directories

- `src/` contains the core code (library, search, training, model etc... implementations).

- `results/` contains results for experiments performed so far as plots or txt files.

- `data/` contains dataset of programs, a stored library as well as stored model weights.

- `scripts/` contains ultility scripts.

### Experiment Scripts

Each of the following scripts can be instructed to perform tasks through the appropriate command line arguments. Use the `-h` flags for script specific details.

- `eval_ohe.py` is a script for training of latent space and transforms based on one-hot representation, followed by evaluation of accuracies and testing of search performance. `eval_ohe_old.py` is an older version which does not make use of parallelism.

- `eval_ohe_transforms_unseen.py` performs experiments on generalization of transforms to unseen shapes/positions in the one hot representation setting. It loads weights for a pretrained encoder and decoder and trains transforms with varying number of shapes/positions present in the training data and plots the results.

- `eval_ohe_img_encoder_unseen.py` performs experiments on generalization of image encoder to unseen shapes in the image input, one-hot output setting. It trains an autoencoder on a one hot representation containing some number of shapes. Then trains a image encoder to target the output of the earlier encoder with some limited set of shapes in the training data. This is then evaluated through search with data containing some other set of shapes. The details regarding the four experiments are present in the notes.

- `eval_image.py` (WORK IN PROGRESS).

## Examples

Complete training of autoencoder and evaluation for one-hot representation and stor results in `results/` and weights in the default location - 
```
python eval_ohe.py --transform_training_epochs 5000 --reconstruction_training_epochs 10000 --eval_n 10 --eval_latent --eval_latent_acc_n 10 --log_path results --save_transforms --save_latent
```

If you dont want to save the weights, then remove the `--save_transforms` and `--save_latent` flags. 

If you dont want to train the networks from scratch but load from the default location then remove the `--trasnform_training_epochs` and `--reconstruction_training_epochs` flags. 

To modify the amount of data being evaluated on, change the numbers in front of the `--eval_n` and `--eval_latent_acc_n` flags.

Similar formats are used for the other scripts. More details for changing random seeds, CPU count, GPU id and other options can be found with the `-h` flag.

## Misc

### Notes

- Reconstruction training (10 000 epochs) takes upto 5 minutes (with GPU)
- Transform training (5000 epochs) takes upto 20 minutes (with GPU) which is cut to 2.5 minutes with parallelism.
- Evaluation (2000 programs) takes upto 15 minutes (with GPU) which is cut down to 2 minutes with parallelism (8 CPUs).
- Unseen shape experiment takes upto 4 hours (with GPU). Which is cut down to 20 minutes with parallelism (8 CPUs).
- Unseen position experiment can take upto 1 hour (with GPU). Which is cut down to 7 minutes with parallelism (8 CPUs).
