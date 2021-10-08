import collections
import copy
import itertools
from itertools import product
import random
from typing import List

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from .image import IMG_SIZE, SHAPE_IMG_SIZE,load_shape_map
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split





BOARD_SIZE = 3


def generate_board_states(
    shapes: List, n_shapes: int = 1, board_size: int = 3, return_final: bool = False
) -> List:
    """Generates data consisting of all possible configuration."""
    positions = list(itertools.product(range(board_size), repeat=2))

    ds = []
    dq = collections.deque()
    dq.append(set())
    while True:
        i = dq.popleft()

        if len(i) == n_shapes:
            dq.appendleft(i)
            break

        for k, s in itertools.product(shapes, positions[len(i) :]):
            j = copy.deepcopy(i)
            j.add((k, s))
            dq.append(j)
            if not return_final:
                ds.append(j)

    if return_final:
        ds = list(dq)

    return ds


def generate_programs_exhaustive(lib, depth: int = 1) -> List:
    """Generates programs of given depth from a given list of transforms"""
    pq = collections.deque()
    pq.append([])

    tf_names = list(lib.primitives_dict.keys())
    conversions = [i for i in tf_names if i.startswith("to-")]

    while True:
        i = pq.popleft()
        if len(i) == depth:
            pq.appendleft(i)
            break

        tf_names_t = tf_names

        if len(i) > 0 and i[-1] in conversions:
            tf_names_t = [k for k in tf_names_t if k not in conversions]

        tf_names_t = (
            tf_names_t + ["out"]
            if (len(i) > 0 and i[-1] in tf_names) or len(i) == 0
            else tf_names_t
        )
        tf_names_t = (
            tf_names_t + ["clear"] if len(i) > 0 and i[-1] == "out" else tf_names_t
        )
        if len(i) == depth - 1:
            if len(i) > 0 and i[-1] in ["out", "clear"]:
                continue
            else:
                tf_names_t = ["out"]

        for tf in tf_names_t:
            c = i.copy()
            c.append(tf)
            pq.append(c)

    return list(pq)


def generate_programs_exhaustive_upto(lib, max_depth: int = 1) -> List:
    ps = []
    for i in range(1, max_depth + 1):
        ps.extend(generate_programs_exhaustive(lib, i))
    return ps


def generate_programs_random(lib, depth: int = 1, num_samples: int = 1) -> List:
    """Generates random programs of given depth from a given list of transforms"""
    ps = []
    n = 0
    w = 0

    tf_names = list(lib.primitives_dict.keys())
    conversions = [i for i in tf_names if i.startswith("to-")]

    while n < num_samples:
        i = []
        drop = False

        for _ in range(depth):
            tf_names_t = tf_names

            if len(i) > 0 and i[-1] in conversions:
                tf_names_t = [k for k in tf_names_t if k not in conversions]

            tf_names_t = (
                tf_names_t + ["out"]
                if (len(i) > 0 and i[-1] in tf_names) or len(i) == 0
                else tf_names_t
            )
            tf_names_t = (
                tf_names_t + ["clear"] if len(i) > 0 and i[-1] == "out" else tf_names_t
            )
            if len(i) == depth - 1:
                if len(i) > 0 and i[-1] in ["out", "clear"]:
                    drop = True
                    continue
                else:
                    tf_names_t = ["out"]

            i.append(random.choice(tf_names_t))

        if not drop and i not in ps:
            ps.append(i)
            n += 1
            w = 0
        elif w == 10:
            n += 1
            w = 0
        else:
            w += 1

    return ps


def generate_programs_random_upto(
    lib, min_depth: int, max_depth: int, num_samples: int
) -> List:
    ps = []
    for i in tqdm(range(min_depth, max_depth + 1)):
        ps.extend(generate_programs_random(lib, i, num_samples))
    return ps


def generate_examples_exhaustive(programs: List, boards: List, lib) -> List:
    """Generates examples consisting of input/output pairs obstained
    by applying a set of given programs to all possible shape_map
    configurations of a given number of shapes on the board
    """
    data = []
    for p, input_board in itertools.product(programs, boards):
        output_board = lib.apply_program(p, input_board)
        if len(output_board) > 0:
            data.append(
                {"input": input_board, "output": output_board, "program": p}
            )
    return data


def generate_examples_random(
    programs: List, boards: List, lib, num_samples: int
) -> List:
    """Generates examples consisting of input/output pairs obstained
    by applying a set of randomly chosen programs to randomly chosen
    configurations of a given number of shapes on the board
    """
    data = []
    num_samples = min(len(programs) * len(boards), num_samples)
    while len(data) < num_samples:
        p = random.choice(programs)
        input_board = random.choice(boards)
        output_board = lib.apply_program(p, input_board)

        if len(output_board) > 0:
            example = {"input": input_board, "output": output_board, "program": p}

            if example not in data:
                data.append(example)
    return data


def generate_random_image_data(n: int, shape_map: dict):
    """Generates given number of random images from shape map"""
    
    random_shapes = random.choices(list(shape_map.keys()), k=n)
    
    limit = (IMG_SIZE[0] - SHAPE_IMG_SIZE[0], IMG_SIZE[1] - SHAPE_IMG_SIZE[1])
    random_positions = [(int(random.random() * limit[0]), int(random.random() * limit[1])) for _ in range(n)]

    images = []    
    for shape, pos in zip(random_shapes, random_positions):
        bg = Image.new("RGB", IMG_SIZE, (255, 255, 255))
        bg.paste(shape_map[shape], box=pos)
        images.append((bg, (shape, pos)))
    
    return images

def get_seen_unseen_split(shape_map: dict,test_size=0.2):
    shapes=list(shape_map.keys())
    seen_shapes,unseen_shapes=train_test_split(shapes,test_size=test_size)
    limit = (IMG_SIZE[0] - SHAPE_IMG_SIZE[0], IMG_SIZE[1] - SHAPE_IMG_SIZE[1])
    positions=[(a,b) for a in range(limit[0]) for b in range(limit[1])]
    seen_pos,unseen_pos=train_test_split(positions,test_size=test_size)
    return seen_shapes,unseen_shapes,seen_pos,unseen_pos

def get_image_data(shapes,positions,shape_map):
    images = []    
    for shape, pos in product(shapes, positions):
        bg = Image.new("RGB", IMG_SIZE, (255, 255, 255))
        bg.paste(shape_map[shape], box=pos)
        images.append((bg, (shape, pos)))
    
    return images


if __name__ == "__main__":
    shape_map = load_shape_map("../data/images")
    seen_shapes,unseen_shapes,seen_pos,unseen_pos=get_seen_unseen_split(shape_map)
    images=get_image_data(seen_shapes,unseen_pos,shape_map)
    print(len(images))
    images[0][0].show()