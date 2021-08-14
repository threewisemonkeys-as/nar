import collections
import copy
import itertools
import random
from typing import List

from .image import IMG_SIZE, SHAPE_IMG_SIZE, load_shape_map

from tqdm import tqdm
from PIL import Image

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
    by applying a set of given programs to all possible shape
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
    
    limit = IMG_SIZE - SHAPE_IMG_SIZE
    random_positions = [(random.random() * limit, random.random() * limit) for _ in range(n)]

    images = []    
    for shape, pos in zip(random_shapes, random_positions):
        bg = Image.new("RGB", IMG_SIZE, (255, 255, 255))
        bg.paste(shape_map[shape], box=pos)
        images.append(bg)
    
    return images


if __name__ == "__main__":
    import pickle

    import dill

    random.seed(42)

    lib = dill.load(open("data/libraries/library0.pkl", "rb"))
    shift_lib = dill.load(open("data/libraries/shift_library0.pkl", "rb"))
    shapes = ["square", "triangle", "circle", "delta"]

    shift_programs_upto_6 = []
    for n in range(1, 7):
        shift_programs_upto_6.append(generate_programs_exhaustive_upto(shift_lib, n))
    pickle.dump(
        shift_programs_upto_6, open("data/programs/shift_programs_upto_6.pkl", "wb")
    )

    programs_upto_20 = []
    for n in range(1, 21):
        programs_upto_20.append(generate_programs_random(lib, n, 1000))
    pickle.dump(programs_upto_20, open("data/programs/programs_upto_20.pkl", "wb"))

    shift_programs_upto_20 = []
    for n in range(1, 21):
        shift_programs_upto_20.append(generate_programs_random(shift_lib, n, 1000))
    pickle.dump(shift_programs_upto_20, open("data/programs/shift_programs_upto_20.pkl", "wb"))

    shape_map = load_shape_map("data/images")
    random_images = generate_random_image_data(1, shape_map)
    random_images[0].save("random_image.png")