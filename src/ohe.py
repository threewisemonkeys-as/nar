import itertools
from typing import Optional

import torch
import torch.nn.functional as F


def ohe_fns_creator(shapes: list, board_size: int = 3):
    """Creates a functions to handle one-hot encoding of board states"""
    positions = list(itertools.product(range(board_size), range(board_size)))
    shapes_map = {shape: i for i, shape in enumerate(shapes)}
    x_map = {x: x + len(shapes) for x in range(3)}
    y_map = {y: y + len(shapes) + 3 for y in range(3)}
    data_split = (len(shapes), board_size, board_size)  # useful later
    N = sum(data_split)

    one_hot_mapping = {}
    for shape, pos in itertools.product(shapes, positions):
        v = [0 for _ in range(N)]
        v[shapes_map[shape]] = 1
        x, y = pos
        v[x_map[x]] = 1
        v[y_map[y]] = 1
        one_hot_mapping[frozenset([(shape, pos)])] = v
        one_hot_mapping[(shape, pos)] = v

    single_object_ohe_hit_check = lambda g, t: [
        i.argmax().item() for i in torch.split(g, data_split, dim=1)
    ] == [i.argmax().item() for i in torch.split(t, data_split, dim=1)]
    one_hot_represent = lambda board: [one_hot_mapping[elem] for elem in board]

    def ohe_decode(ohe_repr):
        decoded = []
        for elem in ohe_repr:
            shape_id, x, y = [
                i.argmax().item() for i in torch.split(elem, data_split, dim=1)
            ]
            shape = shapes[shape_id]
            decoded.append((shape, (x, y)))
        return decoded

    def ohe_hit_check(generated_set, target_set):
        # check if sizes of both sets matches
        if len(generated_set) != len(target_set):
            return False

        # create list of indices of generated
        # set which havent been matched yet
        generated_set = list(generated_set)
        yet_to_match = list(range(len(generated_set)))

        # loop over all elements in the target set
        for i, t in enumerate(target_set):
            found = False

            # check if target element matches any generated one
            for j in yet_to_match:
                if single_object_ohe_hit_check(generated_set[j], t):
                    # if match, then remove index from list of yet to match
                    yet_to_match = [k for k in yet_to_match if k != j]
                    found = True
                    break

            # if no matches found in generated set for
            # current target element, then we have a MISS
            if not found:
                break

        return found

    def ohe_partial_hit_check(generated_set, target_set):
        # check if sizes of both sets matches
        if len(generated_set) > len(target_set):
            return -1

        # create list of indices of target
        # set which havent been matched yet
        target_set = list(target_set)
        yet_to_match = list(range(len(target_set)))

        # loop over all elements in the generated set
        for g in generated_set:
            found = False

            # check if generated element matches any target one
            for j in yet_to_match:
                if single_object_ohe_hit_check(target_set[j], g):
                    # if match, then remove index from list of yet to match
                    yet_to_match = [k for k in yet_to_match if k != j]
                    found = True
                    break

            # if no matches found in target set for current
            # generated element, then we have a MISS
            if not found:
                return -1

        if len(yet_to_match) > 0:
            # partial match if generated set is
            # a subset of target set
            return 0
        else:
            return 1

    def ohe_loss_fn_creator(loss_fns: Optional[list] = None):
        """Creates loss function for one-hot encoded generated and target vectors
        given the chunk sizes for splitting the vector and loss functions to be
        applied to each chunk.

        Required since encoding scheme uses multiple one-hot encodings for
        different aspects (e.g. shape, positions etc...) packed into a single
        vector.
        """
        if loss_fns is None:
            loss_fns = [F.cross_entropy for _ in data_split]

        def ohe_loss_fn(generated, target):
            i = 0
            loss = None
            for k, j in enumerate(data_split):
                if loss is None:
                    loss = loss_fns[k](
                        generated[:, i : i + j], target[:, i : i + j].argmax(1)
                    )
                else:
                    loss += loss_fns[k](
                        generated[:, i : i + j], target[:, i : i + j].argmax(1)
                    )
                i += j
            return loss

        return ohe_loss_fn

    def one_hot_tensor_represent_creator(device: torch.device, dtype: torch.dtype):
        return lambda board: [
            torch.tensor([one_hot_mapping[elem]], device=device, dtype=dtype)
            for elem in board
        ]

    return (
        data_split,
        one_hot_mapping,
        one_hot_represent,
        ohe_decode,
        single_object_ohe_hit_check,
        ohe_hit_check,
        ohe_partial_hit_check,
        ohe_loss_fn_creator,
        one_hot_tensor_represent_creator,
    )


if __name__ == "__main__":
    import dill

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    shapes = ["circle", "square", "triangle", "delta"]
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

    lib = dill.load(open("data/libraries/library0.pkl", "rb"))

    p = ["shiftdown", "out", "clear"]
    input_board = set([("triangle", (1, 1)), ("square", (0, 0))])
    output_board = lib.apply_program(p, input_board)
    input_ohe, output_ohe = [
        one_hot_represent(board) for board in [input_board, output_board]
    ]
    input_ohe_tensor, output_ohe_tensor = [
        one_hot_tensor_represent(board) for board in [input_board, output_board]
    ]
    print("input: ", input_board, input_ohe, input_ohe_tensor)
    print("output: ", output_board, output_ohe, output_ohe_tensor)
