import math
import os.path
import typing
import erutils
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as tf
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from modules.models import ModelConfig
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main(config_path: typing.Union[os.PathLike, str] = 'config/HyperParameters.yaml'):
    config = erutils.read_yaml(config_path)
    img_size = config['img_size']
    hp_generator = erutils.HyperParameters(
        model=config['generator'],
        c_req=config['c_req'], sc=config['zg'],
        detail='Generator Model \n', print_status=True,
        imports=['from modules.modules import * ']
    )
    hp_discriminator = erutils.HyperParameters(
        model=config['discriminator'],
        c_req=config['c_req'], sc=config['zd'],
        detail='Discriminator Model \n', print_status=True,
        imports=['from modules.modules import * ']
    )
    generator = ModelConfig(hp_generator).cuda()
    discriminator = ModelConfig(hp_discriminator).cuda()
    generator.apply(weight_init)
    discriminator.apply(weight_init)
    optimizer_generator = torch.optim.Adam(generator.parameters(), 3e-4, betas=(0.5, 0.99))
    optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), 3e-4, betas=(0.5, 0.99))
    print('Generator Contain ', sum(m.numel() for m in generator.parameters()) / 1e6, ' Million Parameters')
    print('Discriminator Contain ', sum(m.numel() for m in generator.parameters()) / 1e6, ' Million Parameters')

    dataset = ImageFolder(
        root='data/seg_test/seg_test',
        transform=tf.Compose(
            [
                tf.Resize((img_size, img_size)),

                tf.ToTensor(),
                tf.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
            ]
        )
    )
    batch_size = 2

    def __len__():
        return 200

    dataset.__len__ = __len__
    total_iters_pre_ep = math.ceil(dataset.__len__() / batch_size)
    dataloader = DataLoader(dataset=dataset, pin_memory=True, batch_size=batch_size)
    real_label = 1.0
    fake_label = 0.0
    epochs = 40
    tfb = SummaryWriter(log_dir='../out/', filename_suffix='GAN')
    criterion = torch.nn.BCELoss()
    fix_noise = torch.rand((batch_size, hp_generator.sc, 1, 1), dtype=torch.float32).cuda()
    for epoch in range(epochs):
        gl = 0
        dl = 0

        vl = 0
        for i, (x, _) in enumerate(dataloader):
            vl += 1
            x = x.cuda()

            optimizer_discriminator.zero_grad(set_to_none=True)
            noise = torch.rand((batch_size, hp_generator.sc, 1, 1), dtype=torch.float32).cuda()

            fake = generator(noise)

            fake_disc_prediction = discriminator(fake.detach()).view(-1)
            real_disc_prediction = discriminator(x).view(-1)

            fake_loss = criterion(fake_disc_prediction, torch.zeros_like(fake_disc_prediction))
            fake_loss.backward()
            real_loss = criterion(real_disc_prediction, torch.ones_like(real_disc_prediction))
            real_loss.backward()

            dis_loss = real_loss + fake_loss
            dl += dis_loss
            optimizer_discriminator.step()

            optimizer_generator.zero_grad(set_to_none=True)

            fake_prediction = discriminator(fake.detach()).view(-1)
            error_generator = criterion(fake_prediction, torch.ones_like(real_disc_prediction))
            error_generator.backward()
            gl += error_generator
            optimizer_generator.step()
            erutils.fprint(
                f'\rEpoch : [{epoch}/{epochs}] | ITER : {i}/{total_iters_pre_ep} | GENERATOR : {error_generator.item()} | DISCRIMINATOR : {dis_loss.item()}',
                end='')

            # this commet to be raded when ever you tought that yo8u are doin something right exacly for down here optimizer have problems with weifgght decays :))
        print()
        torch.save(dict(
            generator=generator.state_dict(),
            discriminator=discriminator.state_dict(),
            optimizer_generator=optimizer_generator.state_dict(),
            optimizer_discriminator=optimizer_discriminator.state_dict(),
            epoch=epoch,
            epochs=epochs
        ), 'model.pt')
        predicted = generator(fix_noise)
        predicted = (predicted + 1) / 2
        predicted = predicted.detach().cpu()
        image_grid = make_grid(predicted, number_of_rows=batch_size)

        tfb.add_scalar('Train/Loss-Generator', gl / vl, epoch)
        tfb.add_scalar('Train/Loss-Discriminator', dl / vl, epoch)
        tfb.add_image('GENERATED', image_grid)


if __name__ == "__main__":
    main()
