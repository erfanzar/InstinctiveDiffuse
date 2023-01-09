import os.path

import torch.cuda
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from core.discriminator import Discriminator
from core.generator import Generator
from utils.utils import create_noise, calculate_input_dim, weight_init

batch_size = 64
criterion = torch.nn.BCEWithLogitsLoss()
z_dim = 64
lr = 2e-4
display_step = 500
data_shape = 64
number_of_epochs = 200
n_classes = 9
image_size = 416
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":
    data_loader = DataLoader(
        MNIST(
            os.path.join(os.getcwd(), 'MNIST'),
            download=True if not os.path.exists(os.path.join(os.getcwd(), 'MNIST')) else False,

        ), batch_size=batch_size, shuffle=True
    )
    generator_input_dim, discriminator_input_dim = calculate_input_dim(z_dim=z_dim, data_shape=data_shape,
                                                                       n_classes=n_classes)

    generator = Generator('config/hyper-parameters.yaml')
    discriminator = Discriminator('config/hyper-parameters.yaml')

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr)

    generator = generator.apply(weight_init)
    discriminator = discriminator.apply(weight_init)
    # fake_noise = create_noise(batch_size, z_dim, device=device)
    # noise_and_labels = torch.cat((fake_noise.float(), one_hot_labels.float()), 1)
    # fake = generator(noise_and_labels)
