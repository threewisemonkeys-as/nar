import pathlib
import tempfile
import turtle
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.functional import split
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageChops

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


def split_image(img: Image.Image) -> List[Image.Image]:
    """Splits single image with multiple shapes into multiple images with single shapes"""
    gray = np.asarray(img.convert("L"))
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


def load_img(img_path: str) -> Image.Image:
    return Image.open(img_path).convert("L").resize(IMG_SIZE)

def load_img_tensor(img_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(np.asarray(load_img(img_path)), device=device, dtype=dtype).unsqueeze(0) / 255

def raw_img_tensor_represent_creator(device: torch.device, dtype: torch.dtype):
    def raw_img_tensor_represent(img: Image.Image):
        return [
            torch.tensor(np.asarray(elem), device=device, dtype=dtype).unsqueeze(0) / 255
            for elem in split_image(img)
        ]
    return raw_img_tensor_represent

def img_represent_fns_creator(
    shape_map: dict, device: torch.device, dtype: torch.dtype
):

    single_img_represent = lambda board: split_image(draw_board(board, shape_map))

    single_img_tensor_represent = lambda board: [
        torch.tensor(np.asarray(img), device=device, dtype=dtype).unsqueeze(0) / 255
        for img in single_img_represent(board)
    ]

    return single_img_represent, single_img_tensor_represent

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def homography(im1: Image.Image, im2: Image.Image):

    # im1Gray, im2Gray = np.asarray(im1Gray), np.asarray(im2Gray)

    # Convert images to grayscale
    im1Gray = np.asarray(im1.convert("L"))
    im2Gray = np.asarray(im2.convert("L"))

    _,im1Gray = cv2.threshold(im1Gray, 110, 255, cv2.THRESH_BINARY)
    _,im2Gray = cv2.threshold(im1Gray, 110, 255, cv2.THRESH_BINARY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray,None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    print(points1, points2)
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # # Use homography
    # height, width, channels = im2.shape
    # im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return h

def extract_shape_from_image(im: Image.Image) -> Image.Image:
    """Extract crop of input image whih containsthe shape"""
    grid_shape = POS_GRID.shape
    min_v = 255
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            pos = POS_GRID[i, j]
            img_crop = im.crop(
                (
                    pos[0],
                    pos[1],
                    pos[0] + SHAPE_IMG_SIZE[0],
                    pos[1] + SHAPE_IMG_SIZE[1],
                )
            )
            if np.asarray(img_crop).mean() < min_v:
                min_v = np.asarray(img_crop).mean()
                shape_crop = img_crop
    
    return shape_crop


def homography_image_match_creator(thresh: float):
    def homography_image_match(im1, im2):
        im1, im2 = (extract_shape_from_image(i) for i in (im1, im2))
        h = homography(im1, im2)
        v = np.linalg.norm(h - np.eye(*h.shape))
        return v > thresh

    return homography_image_match

def image_equality_check(im1: Image.Image, im2: Image.Image):
    """Checks if two images are exactly the same"""
    # return ImageChops.difference(im1, im2).getbbox() is None
    # Convert images to grayscale
    im1Gray = np.asarray(im1.convert("L"))
    im2Gray = np.asarray(im2.convert("L"))

    _,im1Gray = cv2.threshold(im1Gray, 110, 255, cv2.THRESH_BINARY)
    _,im2Gray = cv2.threshold(im1Gray, 110, 255, cv2.THRESH_BINARY)

    return (im1Gray == im2Gray).all()

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
