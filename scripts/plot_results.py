import torch
import pickle
import io
import wandb
import matplotlib.pyplot as plt
import numpy as np

a = [float(i) for i in open("results/symbolic_latent_unseen_position_accuracies.txt").read().split(',')]

fig = plt.figure()
plt.plot(a)
plt.title("Accuracy for unseen positions")
plt.ylabel("Accuracy")
plt.xlabel("Number of Unseen Positions")
plt.xticks(np.arange(0, 9, 1.0))
plt.yticks(np.arange(0, 1.05, 0.05))
plt.grid()
plt.show()


# api = wandb.Api()
# run = api.run("/atharv/nar/runs/21glvuiu")
# run.log({"unseen_shape_accuracy": fig})

