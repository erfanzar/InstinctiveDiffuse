import os.path

import torch.cuda
import torchvision.datasets as dts
from erutils.utils import read_yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from core.discriminator import Discriminator
from core.generator import Generator
from core.train import train_n

cfg = read_yaml('../config/hyper-parameters.yaml')
batch_size = 128
criterion = torch.nn.BCELoss()
z_dim = cfg['z']
lr = 2e-4
display_step = 10
data_shape = [3, cfg['image_size'], cfg['image_size']]
number_of_epochs = 200
n_classes = cfg['n_classes']
image_size = cfg['image_size']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    criterion = torch.nn.BCEWithLogitsLoss()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # dt = MNIST(
    #     os.path.join(os.getcwd(), 'MNIST'),
    #     download=True if not os.path.exists(os.path.join(os.getcwd(), 'MNIST')) else False,
    #     transform=transform
    # )
    dt = dts.ImageFolder(
        os.path.join(os.getcwd(), 'Data'),
        transform=transform,

    )
    data_loader = DataLoader(
        dt, batch_size=batch_size, shuffle=True
    )

    generator = Generator('config/hyper-parameters.yaml')
    discriminator = Discriminator('config/hyper-parameters.yaml')

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr)

    train_n(generator_model=generator, discriminator_model=discriminator, epochs=number_of_epochs, z_dim=z_dim,
            criterion=criterion, cfg=cfg, image_size=image_size,
            discriminator_optim=discriminator_optimizer, generator_optim=generator_optimizer, batch_size=batch_size,
            device=device, dataloader=data_loader, display_epoch=display_step)
