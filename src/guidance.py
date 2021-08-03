import itertools
from typing import List

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftOracle:
    def __init__(
        self, miss_params=(2, 5), hit_params=(5, 2), distribution=np.random.beta
    ):
        self.miss_params = miss_params
        self.hit_params = hit_params
        self.distribution = distribution

    def __call__(self, c, examples):
        true_program = examples[0]["program"]
        gen_program = c[0]
        if gen_program == true_program[: len(gen_program)]:
            score = self.distribution(*self.hit_params)
        else:
            score = self.distribution(*self.miss_params)
        return score

    def show(self):
        n = 10_000
        data = {
            "hit score": self.distribution(*self.hit_params, size=n),
            "miss score": self.distribution(*self.miss_params, size=n),
        }
        sns.displot(data=data, kind="kde", height=2)


class RuleBasedGuidance:
    def __init__(self, rules, pos_map, noise: float = 0.0):
        self.rules = rules
        self.pos_map = pos_map
        self.noise = noise

    def __call__(self, c, examples):
        score = 0
        k = 0
        tp = c[0]
        for e in examples:
            o_board = e["output"]
            c_board = c[1][k][2]
            k += 1

            count = {
                shape: sum([elem[0] == shape for elem in o_board])
                - sum([elem[0] == shape for elem in c_board])
                for shape in self.rules.shapes
            }
            for s1, s2 in itertools.product(self.ules.shapes, self.rules.shapes):
                if count[s2] > count[s1] and count[s1] != 0:
                    if tp[-1] == f"{s1}-{s2}":
                        score += 0.05

            for e_elem in o_board:
                e_y, e_x = self.pos_map[e_elem[1]]
                t_scores = []

                for c_elem in c_board:
                    if c_elem == e_elem:
                        t_scores.append(0.5)

                    c_y, c_x = self.pos_map[c_elem[1]]

                    if abs(c_y - e_y) == 0 or abs(c_x - e_x) == 0:
                        t_scores.append(0.1)

                    if abs(c_y - e_y) == 1 or abs(c_x - e_x) == 1:
                        t_scores.append(-0.01)

                    if abs(c_y - e_y) == 2 or abs(c_x - e_x) == 2:
                        t_scores.append(-0.02)

                    if abs(c_y - e_y) == 3 or abs(c_x - e_x) == 3:
                        t_scores.append(-0.03)

                score += max(t_scores) if len(t_scores) > 0 else 0.0

        score /= len(examples)
        return np.random.normal(score, self.noise)


class RNNGuidance(nn.Module):
    def __init__(
        self,
        num_primitives: int,
        embed_dim: int = 32,
        num_heads: int = 8,
    ):
        pass


class AttentionGuidance(nn.Module):
    def __init__(
        self,
        num_primitives: int,
        embed_dim: int = 32,
        num_heads: int = 8,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.num_primitives = num_primitives
        self.embed_dim = embed_dim

        self.program_embed_fn = nn.Linear(num_primitives, embed_dim)

        self.io_key_fn = nn.Linear(embed_dim, embed_dim)
        self.io_value_fn = nn.Linear(embed_dim, embed_dim)
        self.token_key_fn = nn.Linear(embed_dim, embed_dim)
        self.token_value_fn = nn.Linear(embed_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.output_fn = nn.Linear(embed_dim, num_primitives)

    def forward(
        self,
        current_program_embeddings: List,
        example_embeddings: List,
    ) -> torch.Tensor:
        """
        Args:
            current_program_embeddings: List of B List(s) of N torch.Tensor(s) each of dim E
            example_embeddings: List of K 2-tuple(s) of torch.Tensor(s) each of dim E

        Returns:
            torch.Tensor:
        """
        # get batch size as number of programs and input length as program length
        B = len(current_program_embeddings)
        N = len(current_program_embeddings[0])

        # stack all input and output embeddings and get keys and values for them
        io = torch.stack(list(itertools.chain(*example_embeddings)))  # 2K, E
        io_key = self.io_key_fn(io).unsqueeze(0).expand(B, -1, -1)  # B, 2K, E
        io_value = self.io_value_fn(io).unsqueeze(0).expand(B, -1, -1)  # B, 2K, E

        # convert program encodings to one hot
        seq = (
            F.one_hot(torch.stack(current_program_embeddings), self.num_primitives)
            .to(self.dtype)
            .to(self.device)
        )  # B, N, number of primitives

        # get keys and values for them
        seq = self.program_embed_fn(seq)  # B, N, E
        seq_key = self.token_key_fn(seq)  # B, N, E
        seq_value = self.token_value_fn(seq)  # B, N, E

        # construct key and value matrices by cat-ing io keys
        # and values with program embeddings
        key = torch.cat([io_key, seq_key], dim=1)  # B, 2K + N, E
        value = torch.cat([io_value, seq_value], dim=1)  # B, 2K + N, E

        # since we want
        query = torch.ones(
            B, 1, self.embed_dim, device=self.device, dtype=self.dtype
        )  # B, 1, E

        # costruct mask
        attn_mask = torch.ones(1, N + 2, device=self.device, dtype=self.dtype)

        # apply attention
        x, _ = self.attn(query, key, value, attn_mask=attn_mask)  # B, 1, E

        # process output
        x = x.reshape(B, self.embed_dim)  # B, E
        x = self.output_fn(F.relu(x))  # B, number of primitives (logits)

        return x


if __name__ == "__main__":

    from image import single_img_tensor_represent
    from models import ConvDecoder, ConvEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    embedding_dim = 32
    shift_tf_names = ["shiftright", "shiftdown", "shiftup", "shiftleft"]
    io_embedding = ConvEncoder(img_size, embedding_dim, True).to(device).to(dtype)
    tf_id_map = {tf: id + 2 for id, tf in enumerate(shift_tf_names)}
    tf_id_map["clear"] = 0
    tf_id_map["out"] = 1
    program_embedding = lambda x: torch.tensor(
        [tf_id_map[tf] for tf in x], dtype=torch.long, device=device
    )

    te = [
        x
        for x in upto_twenty_level_one_shape_examples
        if all([y in shift_tf_names + ["out", "clear"] for y in x["program"]])
    ]
    tei = te[500]
    print(tei)

    model = AttentionGuidance(6, embedding_dim, 8).to(device).to(dtype)
    pe = program_embedding(tei["program"])
    r = model(
        [pe[:-1]],
        [
            io_embedding(single_img_tensor_represent(tei[k])[0])
            for k in ["input", "output"]
        ],
    )
    print(r.shape, pe[:-1].shape, pe[-1].unsqueeze(0).shape)
    F.nll_loss(r, pe[-1].unsqueeze(0))
