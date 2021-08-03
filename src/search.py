import collections
import time
from typing import Optional

import multiprocess as mp
import torch.nn as nn
from tqdm import tqdm


def exhaustive_search_creator(hit_check, max_depth: int = None, timeout: int = None):
    """Creates a search function given a way to check hits
    and maximum search depth and timeout (whichever expired first).
    """
    assert not (
        max_depth is None and timeout is None
    ), "must specify either timeout or max_depth"

    def exhaustive_search(
        examples,
        input_repr_encoder,
        output_repr_encoder,
        encoder,
        decoder,
        transforms,
        apply_transform,
        input_repr_decoder=None,
        output_repr_decoder=None,
    ):
        start_time = time.time()

        n_examples = len(examples)

        # set input (i) and as encoding of given inputs
        i = [
            [encoder(elem) for elem in input_repr_encoder(e[f"input"])]
            for e in examples
        ]

        # extract list of outputs
        o = [output_repr_encoder(e[f"output"]) for e in examples]

        # create deque to store frontier and add empty element
        # with memory intialised to input
        pq = collections.deque()
        pq.append([[], [[i_k.copy(), [], []] for i_k in i]])

        # list of transforms used to extend the search
        tfs = list(transforms.keys()) + ["out", "clear"]

        while True:
            # pop from frontier
            h = pq.popleft()

            # if max length reached then exit
            if max_depth is not None:
                if len(h[0]) == max_depth:
                    result = "max_depth"
                    break

            # if timeout is reached then exit
            if timeout is not None:
                if (time.time() - start_time) > timeout:
                    result = "timeout"
                    break

            # extend for each transforms
            for tf in tfs:

                # create new element from popped element
                c = [h[0].copy(), [[h_k_i.copy() for h_k_i in h_k] for h_k in h[1]]]

                # append transform to program of current element
                c[0].append(tf)

                # if transform is out, then decode things
                # in the memory and move to output
                if tf == "out":
                    for k in range(n_examples):
                        new = [decoder(e) for e in c[1][k][0]]
                        c[1][k][1] += new
                        if output_repr_decoder is not None:
                            c[1][k][2] += output_repr_decoder(new)

                # if transform is clear, then move input to memory
                elif tf == "clear":
                    for k in range(n_examples):
                        c[1][k][0] = i[k].copy()

                # otherwise, execute transform on memory
                else:
                    # get the N.N. (or vector) for this transform
                    tf_val = transforms[tf]

                    # apply it to each element in the memory
                    for k in range(n_examples):
                        c[1][k][0] = [
                            apply_transform(tf_val, elem) for elem in c[1][k][0]
                        ]

                # check if we have a hit.
                if all([hit_check(c[1][k][1], o[k]) for k in range(n_examples)]):
                    return "success", c, time.time() - start_time

                # if we dont have a hit, then add to frontier and carry on search
                pq.append(c)

            # remove previously popped element from memory
            del h
        return result, None, time.time() - start_time

    return exhaustive_search


def pruned_search_creator(
    single_object_hit_check, max_depth: int = None, timeout: int = None
):
    """Creates a search function given a way to check hits
    and maximum search depth.
    """

    assert not (
        max_depth is None and timeout is None
    ), "must specify either timeout or max_depth"

    def pruned_search(
        examples,
        input_repr_encoder,
        output_repr_encoder,
        encoder,
        decoder,
        transforms,
        apply_transform,
        input_repr_decoder=None,
        output_repr_decoder=None,
    ):
        start_time = time.time()

        n_examples = len(examples)

        # set input (i) and as encoding of given inputs
        i = [
            [encoder(elem) for elem in input_repr_encoder(e[f"input"])]
            for e in examples
        ]

        # extract list of outputs
        o = [output_repr_encoder(e[f"output"]) for e in examples]

        # create deque to store frontier and add empty element
        # with memory intialised to input
        pq = collections.deque()
        pq.append([[], [[i[k].copy(), [], [], o[k].copy()] for k in range(n_examples)]])
        # node: [program, [(input, memory, output, left_to_match) for each example] ]

        min_lens = [len(o_k) for o_k in o]

        # list of transforms used to extend the search
        tfs = list(transforms.keys()) + ["out", "clear"]

        while True:
            # pop from frontier
            h = pq.popleft()

            # if max length reached then exit
            if max_depth is not None:
                if len(h[0]) == max_depth:
                    result = "max_depth"
                    break

            # if timeout is reached then exit
            if timeout is not None:
                if (time.time() - start_time) > timeout:
                    result = "timeout"
                    break

            # prune branches not reducing target set
            if any([len(h[1][k][3]) > min_lens[k] for k in range(n_examples)]):
                t_tfs = ["out"]
            else:
                t_tfs = tfs

            # extend for each transforms
            for tf in t_tfs:

                # create new element from popped element
                c = [h[0].copy(), [[h_k_i.copy() for h_k_i in h_k] for h_k in h[1]]]

                # append transform to program of current element
                c[0].append(tf)

                match_elem = None

                # if transform is out, then decode things
                # in the memory and move to output
                if tf == "out":
                    for k in range(n_examples):
                        new = [decoder(e) for e in c[1][k][0]]
                        c[1][k][1] += new

                        # check if generated set is still subset of target set
                        if len(c[1][k][1]) > len(o[k]):
                            continue
                        for g in new:
                            match_elem = False
                            for ti, t in enumerate(c[1][k][3]):
                                if single_object_hit_check(t, g):
                                    match_elem = True
                                    c[1][k][3].pop(ti)
                                    break
                            if not match_elem:
                                break

                        if not match_elem:
                            break

                        if len(c[1][k][3]) == 0:
                            return "success", c, time.time() - start_time

                        # store output representation if function is specified
                        if output_repr_decoder is not None:
                            c[1][k][2] += output_repr_decoder(new)

                    # update minimum size of target set
                    c_lens = [len(c[1][k][3]) for k in range(n_examples)]
                    if all([c_lens[k] <= min_lens[k] for k in range(n_examples)]):
                        min_lens = c_lens

                # if transform is clear, then move input to memory
                elif tf == "clear":
                    for k in range(n_examples):
                        c[1][k][0] = i[k].copy()

                # otherwise, execute transform on memory
                else:
                    # get the N.N. (or vector) for this transform
                    tf_val = transforms[tf]

                    # apply it to each element in the memory
                    for k in range(n_examples):
                        c[1][k][0] = [
                            apply_transform(tf_val, elem) for elem in c[1][k][0]
                        ]

                if match_elem is None or match_elem is True:
                    pq.append(c)

            del h
        return result, None, time.time() - start_time

    return pruned_search


def guided_beam_search_creator(
    guidance_model,
    trim_func,
    hit_check,
    beam_width: int,
    max_depth: int = None,
    timeout: int = None,
):
    """Creates a guided beam search function given a guidance model, a way to
    trim the beam, a way to check hits, beam width and maximum search depth.
    """

    assert not (
        max_depth is None and timeout is None
    ), "must specify either timeout or max_depth"

    def beam_search(
        examples,
        input_repr_fn,
        output_repr_fn,
        encoder,
        decoder,
        transforms,
        apply_transform,
        input_repr_decoder=None,
        output_repr_decoder=None,
    ):

        start_time = time.time()

        n_examples = len(examples)

        # set input (i) and as encoding of given inputs
        i = [[encoder(elem) for elem in input_repr_fn(e[f"input"])] for e in examples]

        # extract list of outputs
        o = [output_repr_fn(e[f"output"]) for e in examples]

        # create empty beam and add empty element with memory intialised to input
        beam = []
        beam.append([[], [[i_k.copy(), [], []] for i_k in i]])

        # list of transforms used to extend the search
        tfs = list(transforms.keys()) + ["out", "clear"]

        # initialised depth count to 0
        d = 0
        while True:
            # initialise new beam
            new_beam = []

            # iterate over each program in beam
            for h in beam:

                # extend for each transforms
                for tf in tfs:

                    # create new element from popped element
                    c = [h[0].copy(), [[h_k_i.copy() for h_k_i in h_k] for h_k in h[1]]]

                    # append transform to program of current element
                    c[0].append(tf)

                    # if transform is out, then decode things
                    # in the memory and move to output
                    if tf == "out":
                        for k in range(n_examples):
                            new = [decoder(e) for e in c[1][k][0]]
                            c[1][k][1] += new
                            if output_repr_decoder is not None:
                                c[1][k][2] += output_repr_decoder(new)

                    # if transform is clear, then move input to memory
                    elif tf == "clear":
                        for k in range(n_examples):
                            c[1][k][0] = i[k].copy()

                    # otherwise, execute transform on memory
                    else:
                        # get the N.N. (or vector) for this transform
                        tf_val = transforms[tf]

                        # apply it to each element in the memory
                        for k in range(n_examples):
                            c[1][k][0] = [
                                apply_transform(tf_val, elem) for elem in c[1][k][0]
                            ]

                    # check if we have a hit.
                    if all([hit_check(c[1][k][1], o[k]) for k in range(n_examples)]):
                        return "success", c, time.time() - start_time

                    # if we dont have a hit, then add to frontier and carry on search
                    new_beam.append(c)

            # after extending all programs in beam, trim it -
            beam = trim_func(new_beam, guidance_model, beam_width, examples)
            # print([c[0] for c in beam])

            # increment depth count
            d += 1

            if d > max_depth:
                result = "max_depth"
                break

            if (time.time() - start_time) > timeout:
                result = "timeout"
                break

        return result, None, time.time() - start_time

    return beam_search


def select_top_k(candidates, model, k, examples):
    scores = [model(c, examples) for c in candidates]
    sorted_candidates = [
        x
        for x, _ in sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )
    ]
    return sorted_candidates[:k]


import dill


def run_dill_encoded(args, dilled_args):
    fun, undilled_args = dill.loads(dilled_args)
    return fun(**args, **undilled_args)


def apply_async(pool, func, args, args_to_dill):
    dilled_args = dill.dumps((func, args_to_dill))
    return pool.apply_async(run_dill_encoded, (args, dilled_args))


def search_test(
    data: dict,
    encoder: nn.Module,
    decoder: nn.Module,
    input_repr_encoder: callable,
    output_repr_encoder: callable,
    transforms: dict,
    apply_transform: callable,
    search: callable,
    input_repr_decoder: Optional[callable] = None,
    output_repr_decoder: Optional[callable] = None,
    n_workers: Optional[int] = None,
):
    """Tests how well a search strategy, an encoder decoder pair and a set of
    transforms perform in terms of accuracy on given set of examples
    """
    hits = 0
    timeouts = 0
    max_depths = 0
    results = {"details": []}

    if n_workers is None:

        # for each example in data search for a program that satisfies the example
        for i, example in enumerate(tqdm(data)):
            result, solution, time_taken = search(
                [example],
                input_repr_encoder,
                output_repr_encoder,
                encoder,
                decoder,
                transforms,
                apply_transform,
                input_repr_decoder,
                output_repr_decoder,
            )

            # update results
            r = {}
            r["example"] = example
            r["result"] = result
            r["time"] = time_taken
            r["solution"] = solution
            results["details"].append(r)

            if result == "success":
                hits += 1
            elif result == "timeout":
                timeouts += 1
            elif result == "max_depth":
                max_depths += 1

    else:
        pbar = tqdm(total=len(data))
        pbar_update = lambda *a: pbar.update()

        pool = mp.Pool(processes=n_workers)
        jobs = []
        # for each example in data search for a program that satisfies the example
        for example in data:
            jobs.append(
                pool.apply_async(
                    search,
                    args=(
                        [example],
                        input_repr_encoder,
                        output_repr_encoder,
                        encoder,
                        decoder,
                        transforms,
                        apply_transform,
                        input_repr_decoder,
                        output_repr_decoder,
                    ),
                    callback=pbar_update,
                )
            )

        for j in jobs:
            result, solution, time_taken = j.get()

            # update results
            r = {}
            r["example"] = example
            r["result"] = result
            r["time"] = time_taken
            r["solution"] = solution
            results["details"].append(r)

            if result == "success":
                hits += 1
            elif result == "timeout":
                timeouts += 1
            elif result == "max_depth":
                max_depths += 1

        pool.close()
        pool.terminate()
        pool.join()

    results["summary"] = {}
    results["summary"]["hit_rate"] = hits / len(data)
    results["summary"]["timeout_rate"] = timeouts / len(data)
    results["summary"]["max_depth_rate"] = max_depths / len(data)

    return results
