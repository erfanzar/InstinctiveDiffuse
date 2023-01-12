import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from engine.commons import *


def create_noise(n_sample: int, input_dim: int, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    return torch.randn(n_sample, input_dim, device=device)


def plot_image_from_tensor(tensor, num_images: int = 30, size: tuple = (1, 416, 416), number_of_rows: int = 5,
                           show: bool = True):
    tensor = (tensor + 1) / 2
    image_unflat = tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], number_of_rows=number_of_rows)

    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
    else:
        return image_grid.permute(1, 2, 0).squeeze()


def weight_init(m):
    if isinstance(m, GeneratorBlock) or isinstance(m, DiscriminatorBlock):
        # print(m.m[0].conv)
        torch.nn.init.normal_(m.m[0].conv.weight, 0.0, 0.2)
        if m.fl is False:
            torch.nn.init.normal_(m.m[0].batch_norm.weight, 0.0, 0.2)


def ohn_vector_from_labels(labels, n_classes):
    return torch.nn.functional.one_hot(labels, num_classes=n_classes)


def calculate_input_dim(z_dim, data_shape, n_classes):
    """

    :param z_dim: dimensions of z
    :param data_shape: example x.shape
    :param n_classes: number of classes
    :return: generator_input_dim, discriminator_input_dim
    """
    generator_input_dim = z_dim + n_classes
    discriminator_input_dim = data_shape[0] + n_classes
    return generator_input_dim, discriminator_input_dim
