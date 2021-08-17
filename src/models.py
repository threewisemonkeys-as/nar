from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Optional[List[int]] = [32, 32, 32],
    activation: Optional[nn.Module] = nn.ReLU(),
) -> nn.Sequential:
    """Utility function for creating custom MLPs made up of linear layers
    Args:
        input_dim (int): length of input vector
        output_dim (int): length of output vector
        hidden_dims (Optional[list[int]], optional): list of dimensions of hidden layers.
            Defaults to [32, 32, 32].
        activation (Optional[nn.Module], optional): activation function to apply between each linear layer.
            Defaults to nn.ReLU().
    Returns:
        nn.Sequential: specified mlp as nn.Module
    """
    modules = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for i in range(len(hidden_dims) - 1):
        modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        modules.append(activation)
    modules.append(nn.Linear(hidden_dims[-1], output_dim))
    # return torch.jit.script(nn.Sequential(*modules))
    return nn.Sequential(*modules)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        latent_dim: int,
        input_is_2d: bool = False,
        maxpool: bool = False,
    ):
        """Encoder to transform input image to representation space
        Args:
            input_shape (tuple[int, int]): shape of input image tensors
            latent_dim (int): length of encoding vector
        """
        super().__init__()

        assert (len(input_shape) == 3 and not input_is_2d) or (
            len(input_shape) == 2 and input_is_2d
        ), "shape not supported"

        self._2d_input = input_is_2d
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # double number of channels and half image dimensions at each
        # convolution with kernel size = 3, stride = 2, padding = 1
        min_dim, max_dim = min(input_shape[-2:]), max(input_shape[-2:])
        num_c = 1 if self._2d_input else input_shape[0]

        self._conv = [nn.Conv2d(num_c, 4, 3, 2, 1), nn.ReLU()]
        num_c = 4
        min_dim, max_dim = [int((x + 1) / 2) for x in [min_dim, max_dim]]

        while min_dim != 1:
            self._conv.append(nn.Conv2d(num_c, 2 * num_c, 3, 2, 1))
            if maxpool:
                self._conv.append(nn.MaxPool2d(3, 2, 1))
            self._conv.append(nn.ReLU())
            num_c *= 2
            min_dim, max_dim = [int((x + 1) / 2) for x in [min_dim, max_dim]]
            if maxpool:
                min_dim, max_dim = [int((x + 1) / 2) for x in [min_dim, max_dim]]

        self._conv = nn.Sequential(*self._conv)
        self._linear = nn.Linear(num_c * min_dim * max_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms image into encoded vector
        Args:
            x (torch.Tensor): An image as tensor
        Returns:
            torch.Tensor: embedding vector
        """
        x = x.unsqueeze(-3) if self._2d_input else x
        x = self._conv(x)
        x = self._linear(x.view(*x.shape[:-2]))
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self,
        output_shape: List[int],
        latent_dim: int,
        output_is_2d: bool = False,
        upsample: bool = False,
    ):
        """Encoder to transform input image to representation space
        Args:
            input_shape (tuple[int, int]): shape of input image tensors
            latent_dim (int): length of encoding vector
        """
        super().__init__()

        assert (len(output_shape) == 3 and not output_is_2d) or (
            len(output_shape) == 2 and output_is_2d
        ), "shape not supported"

        self._2d_output = output_is_2d
        self.output_shape = output_shape
        self.latent_dim = latent_dim

        min_dim, max_dim = min(output_shape[-2:]), max(output_shape[-2:])
        pre_conv_dim = latent_dim if upsample else min_dim * 2
        self._linear = nn.Linear(latent_dim, pre_conv_dim)

        # half number of channels and double image dimensions at each
        # convolution with kernel size = 3, stride = 2, padding = 1
        self._conv = []
        out_c = 1 if self._2d_output else output_shape[0]
        num_c = pre_conv_dim
        dim = 1

        # while dim <= (min_dim // 2):
        while num_c > 4:
            self._conv.append(nn.ConvTranspose2d(num_c, num_c // 2, 3, 2, 1, 1))
            if upsample:
                self._conv.append(nn.Upsample(scale_factor=2))
            self._conv.append(nn.ReLU())
            num_c //= 2
            dim *= 2
            if upsample:
                dim *= 2
        self._conv.append(nn.ConvTranspose2d(num_c, out_c, 3, 2, 1, 1))
        dim *= 2
        if dim == min_dim * 2:
            self._conv.append(nn.Conv2d(out_c, out_c, 3, 2, 1))
        elif dim == min_dim:
            self._conv.append(nn.Conv2d(out_c, out_c, 3, 1, 1))
        else:
            raise Exception("Cant work")
        self._conv = nn.Sequential(*self._conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms image into encoded vector
        Args:
            x (torch.Tensor): An image as tensor
        Returns:
            torch.Tensor: embedding vector
        """
        x = F.relu(self._linear(x))
        x = x.view(*x.shape[:-1], -1, 1, 1)
        x = self._conv(x)
        x = x.squeeze(-3) if self._2d_output else x
        return x


apply_nn_transform = lambda tf, elem: tf(elem)


if __name__ == "__main__":
    import pickle

    from image import IMG_SIZE, draw_board, img_represent_fns_creator, load_shape_map
    from utils import count_parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    boards = pickle.load(open("data/boards/boards_upto_3.pkl", "rb"))
    shape_map = load_shape_map("data/images")
    single_img_represent, single_img_tensor_represent = img_represent_fns_creator(
        shape_map, device, dtype
    )

    board = boards[0][0]
    print(board)
    draw_board(board, shape_map)

    # hyper parameters
    input_shape = IMG_SIZE
    latent_dim = 32
    encoder_lr = 1e-3
    decoder_lr = 1e-3
    tf_lr = 1e-3

    # instantiate nets
    img_encoder = ConvEncoder(input_shape, latent_dim, True).to(device).to(dtype)
    img_decoder = ConvDecoder(input_shape, latent_dim, True).to(device).to(dtype)

    img = single_img_tensor_represent(board)[0]

    enc = img_encoder(img)
    dec = img_decoder(enc)
    print(img.shape, enc.shape, dec.shape)
    print(img_encoder, "\n", img_decoder)
    count_parameters(img_encoder)
    count_parameters(img_decoder)
