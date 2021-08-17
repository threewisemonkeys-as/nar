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

- `scripts/` contains utility scripts.

- `examples/` contains data for examples.

### Experiment Scripts

Each of the following scripts can be instructed to perform tasks through the appropriate command line arguments. Use the `-h` flags for script specific details.

- `eval_ohe.py` is a script for training of latent space and transforms based on one-hot representation, followed by evaluation of accuracies and testing of search performance. `eval_ohe_old.py` is an older version which does not make use of parallelism.

- `eval_ohe_transforms_unseen.py` performs experiments on generalization of transforms to unseen shapes/positions in the one hot representation setting. It loads weights for a pretrained encoder and decoder and trains transforms with varying number of shapes/positions present in the training data and plots the results.

- `eval_ohe_img_encoder_unseen.py` performs experiments on generalization of image encoder to unseen shapes in the image input, one-hot output setting. It trains an autoencoder on a one hot representation containing some number of shapes. Then trains a image encoder to target the output of the earlier encoder with some limited set of shapes in the training data. This is then evaluated through search with data containing some other set of shapes. The details regarding the four experiments are present in the notes.

- `eval_image.py` is similar to `eval_ohe.py` but for a setting with image encoder and decoder. It provides 3 main options for training the image autoencoder - (1) directly with data being all possible combination of shapes and positions, (2) directly with data being shapes randomly places on the board (not resricted to a grid), (3) First trainin a ohe-encoder and image-decoder as an autoencoder and then freezing the decoder weights and training the image encoder. It also provides an option to train transforms and evaluate the system.

- `image2ohe_e2e.py` is a script for solving a task in an end to end manner using the symbolic system. The script requires the paths to encoder, decoder and transforms as well as a directory containin the task. This task directory must contain a `spec.yml` file with the specification of the task and the relevent images. The script searches for the correct program, if found it applies this program to the specified query image and displays the result. Check `examples/simple_shift_task/` for an example of a task.


### Stored Data

#### `data/images`
Images of 20 shapes used to contruct boards.

#### `data/libraries`

Pickles of the full library (`library.pkl`) with both shift and to-shape transforms along with only shift transform library (`shift_library0.pkl`). Code to create these can be found in `src/library.py`

#### `data/programs`

Datasets of programs. Code to create these can be found in `src/datagen.py`

- `programs_upto_20.pkl`: List of 1000 programs for each program length upto 20. (Redundant programs are removed so that lower length programs may have less that 1000)
- `shift_programs_upto_20.pkl`: Similar to above but with only shift transforms
- `shift_programs_upto_6.pkl`:  List of all possible programs for each program length upto 6 with only shift transforms.


#### `data/weights`

Saved model weights.
 
- `simple_encoder.pkl`, `simple_decoder.pkl`,  `simple_transforms.pkl`: Autoencoder and transforms trained over one-hot representation using `eval_ohe.py` with 20 _seen_ shapes (so 20 slots for shapes in the one-hot).
- `simple_encoder_unseen.pkl`, `simple_decoder_unseen.pkl`,  `simple_transforms_unseen.pkl`: Same as above but with all shapes being unseen (no unique slots for shapes in one-hot).

- `img_encoder.pkl`, `img_decoder.pkl`: The encoder in this pair is trained with `eval_ohe_img_encoder.py` as a CNN which takes in images as input and targets the latent space of (`simple_encoder.pkl`, `simple_decoder.pkl`). The decoder is trained to produce images from this latent space.
- `img_encoder_unseen.pkl`, `img_decoder_unseen.pkl`:  The same as above but the image encoder and decoder are trained to target latent space of (`simple_encoder_unseen.pkl`, `simple_decoder_unseen.pkl`) where all shapes are considered unseen.

- `img_encoder_full.pkl`, `img_decoder_full.pkl`,  `img_transforms.pkl`: Autoencoder and transforms trained over images using `eval_img.py` using direct reconstruction training.
- `img_encoder2_full.pkl`, `img_decoder2_full.pkl`,  `img_transforms2.pkl`: Same as above but these are trained by first training an ohe-encoder with an image-decoder, then freezing the image-decoder and training an image-encoder with it.


- `ohe_img_encoder_unseen/`: Directory containing one-hot encoders, one-hot decoders, image encoders and transforms for _shapes unseen to latent space_ experiments.

## Examples

### Training and evaluation

Complete training of autoencoder and evaluation for one-hot representation and stor results in `results/` and weights in the default location -

```
python eval_ohe.py --transform_training_epochs 5000 --reconstruction_training_epochs 10000 --eval_n 10 --eval_latent --eval_latent_acc_n 10 --log_path results --save_transforms --save_latent
```

If you dont want to save the weights, then remove the `--save_transforms` and `--save_latent` flags.

If you dont want to train the networks from scratch but load from the default location then remove the `--trasnform_training_epochs` and `--reconstruction_training_epochs` flags.

To modify the amount of data being evaluated on, change the numbers in front of the `--eval_n` and `--eval_latent_acc_n` flags.

Similar formats are used for the other scripts. More details for changing random seeds, CPU count, GPU id and other options can be found with the `-h` flag.

### End to end evaluation

```
python image2ohe_e2e.py --task_dir examples/simple_shift_task/ --encoder data/weights/img_encoder_unseen.pkl --decoder data/weights/simple_decoder_unseen.pkl --transforms data/weights/simple_transforms_unseen.pkl --max_depth 5 
```
