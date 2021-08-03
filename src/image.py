import pathlib
import tempfile
import turtle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

# def draw_triangle(size: int):
#     """Draw triangle of given size as side length with turtle"""


def poly(tur, shape, quad=(0, 0), size=50):
    tur.penup()
    tur.setpos(quad[0], quad[1])
    tur.setheading(0)
    tur.width(5)
    if shape == "triangle":
        tur.right(90)
        tur.forward(size / 3)
        tur.left(90)
        tur.backward(size / 2)
        k = 3
        turn = 120
    if shape == "square":
        tur.right(90)
        tur.forward(size / 2)
        tur.left(90)
        tur.backward(size / 2)
        k = 4
        turn = 90
    if shape == "circle":
        tur.right(90)
        tur.forward(size / 2)
        tur.left(90)
    if shape == "delta":
        tur.left(90)
        tur.forward(size / 3)
        tur.right(90)
        tur.backward(size / 2)
        k = 3
        turn = -120
    tur.pendown()
    if shape == "circle":
        tur.circle(size / 2)
    else:
        for i in range(k):
            tur.forward(size)
            tur.left(turn)


def draw_simple_shape(shape):
    """Draws simple shape with turtle"""
    tur = turtle.Turtle(visible=False)
    turtle.resetscreen()
    turtle.setup(100, 100)
    turtle.speed(speed=0)
    for t in turtle.turtles():
        t.ht()
    turtle.tracer(0, 0)

    poly(tur, shape)

    turtle.update()
    fp = tempfile.NamedTemporaryFile(suffix=".ps", prefix="turtle_drawing_")
    tur.hideturtle()
    for t in turtle.turtles():
        t.ht()
    tur.getscreen().getcanvas().postscript(file=fp.name)
    return Image.open(fp.name)


POS_GRID = np.array(
    [
        [(2, 2), (22, 2), (42, 2)],
        [(2, 22), (22, 22), (42, 22)],
        [(2, 42), (22, 42), (42, 42)],
    ]
)
IMG_SIZE = (64, 64)
SHAPE_IMG_SIZE = (20, 20)


def draw_board(board, shape_map: dict):
    bg = Image.new("RGB", IMG_SIZE, (255, 255, 255))
    for elem in board:
        bg.paste(shape_map[elem[0]], box=tuple(POS_GRID[elem[1]]))
    return bg


def draw_tensor_board(tensor_board):
    s = np.asarray(sum(tensor_board).squeeze(0).detach().cpu())
    plt.figure()
    plt.imshow(s, cmap="gray")


def draw_examples(examples):
    n = len(examples)
    w, h = 135, 135
    grid = Image.new("RGB", size=(2 * w, n * h))
    for i, e in enumerate(examples):
        grid.paste(
            ImageOps.expand(draw_board(e["input"]), border=1, fill="black"),
            box=(0, i * h),
        )
        grid.paste(
            ImageOps.expand(draw_board(e["output"]), border=1, fill="black"),
            box=(w, i * h),
        )
    return grid


def split_image(img: Image.Image):
    """Splits single image with multiple shapes into multiple images with single shapes"""
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    binary = cv2.bitwise_not(binary)
    ret, labels = cv2.connectedComponents(binary)

    output = []
    for label in range(1, ret):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255
        output.append(Image.fromarray(cv2.bitwise_not(mask)))

    return output


def load_shape_map(save_dir: str) -> dict:
    """Loads PIL images into a dictionary with shapes as keys"""
    save_dir = pathlib.Path(save_dir)
    img_dict = {}
    for img_path in save_dir.glob("*.png"):
        img_dict[img_path.stem] = (
            Image.open(img_path).convert("L").resize(SHAPE_IMG_SIZE)
        )
    return img_dict


def img_represent_fns_creator(
    shape_map: dict, device: torch.device, dtype: torch.dtype
):

    single_img_represent = lambda board: split_image(draw_board(board, shape_map))

    single_img_tensor_represent = lambda board: [
        torch.tensor(np.asarray(img), device=device, dtype=dtype).unsqueeze(0) / 255
        for img in single_img_represent(board)
    ]

    return single_img_represent, single_img_tensor_represent


single_object_img_hit_check_creator = (
    lambda thresh: lambda i1, i2: F.mse_loss(i1, i2) <= thresh
)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    for shape in ["square", "square", "triangle", "circle", "delta"]:
        shape_img = draw_simple_shape(shape)
        shape_img.save(f"data/images/{shape}.png")

    shape_map = load_shape_map("data/images")

    board = [("triangle", (0, 0)), ("circle", (1, 1)), ("x", (2, 2))]
    draw_board(board, shape_map).show()

    single_img_represent, single_img_tensor_represent = img_represent_fns_creator(
        shape_map, device, dtype
    )

    board_img_tensors = single_img_tensor_represent(board)
    fig, axs = plt.subplots(1, 3)
    for n in range(len(board)):
        axs[n].imshow(np.asarray(board_img_tensors[n].squeeze(0).cpu()), cmap="gray")
    plt.show()
