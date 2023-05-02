from jax.experimental.stax import Conv, Dense, Relu, BatchNorm, Flatten, LogSoftmax
from jax.experimental import stax

from utils.utils import *

# the code should be safely removed not used in baseline

# def train_c(epochs: int = 500, generator_model: torch.nn.Module = None, discriminator_model: torch.nn.Module = None,
#             generator_optim: torch.optim.Adam = None,
#             discriminator_optim: torch.optim.Adam = None,
#             use_init_weight: bool = True, lr: float = 2e-4, image_size: int = 28,
#             data_shape: typing.Union[tuple, list] = None, cfg=None,
#             display_epoch: int = 50, criterion: torch.nn = None, n_classes: int = 10,
#             device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu', z_dim: int = 240, batch_size: int = 128,
#             dataloader=None):
#     generator = generator_model.apply(weight_init)
#     discriminator = discriminator_model.apply(weight_init)
#     real_label = 1
#     fake_label = 0
#     generator_input_dim, discriminator_input_dim = calculate_input_dim(z_dim=z_dim, data_shape=data_shape,
#                                                                        n_classes=n_classes)
#     discriminator_losses = []
#     generator_losses = []
#     for epoch in range(epochs):
#         for index, (real_x, labels) in enumerate(dataloader):
#             discriminator_optim.zero_grad()
#
#             real_x = real_x.to(device)
#             labels = labels.to(device)
#             b_size = real_x.cpu().size(0)
#
#             one_hot_labels = ohn_vector_from_labels(labels=labels, n_classes=n_classes)
#             image_one_hot_labels = one_hot_labels[:, :, None, None].repeat(1, 1, image_size, image_size)
#             # attar_print(image_one_hot_labels=image_one_hot_labels.shape)
#             fake_noise = create_noise(len(real_x), input_dim=z_dim, device=device)
#             noise_and_labels = torch.concat((fake_noise, one_hot_labels), 1)
#             # attar_print(noise_and_labels=noise_and_labels.shape)
#             # print(f'noise_and_labels : {noise_and_labels}')
#             fake = generator(noise_and_labels)
#
#             # attar_print(fake=fake.shape)
#             # attar_print(fake_noise=fake_noise.shape)
#
#             assert len(fake) == len(real_x)
#
#             fake_image_and_labels = torch.cat((fake, image_one_hot_labels), 1)
#             real_image_and_labels = torch.cat((real_x, image_one_hot_labels), 1)
#
#             discriminator_fake_pred = discriminator(fake_image_and_labels.detach())
#             discriminator_real_pred = discriminator(real_image_and_labels)
#             assert len(discriminator_real_pred) == len(real_x)
#             assert torch.any(fake_image_and_labels != real_image_and_labels)
#
#             discriminator_fake_loss = criterion(discriminator_fake_pred, torch.zeros_like(discriminator_fake_pred))
#             discriminator_real_loss = criterion(discriminator_real_pred, torch.ones_like(discriminator_real_pred))
#             discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
#             discriminator_real_loss.backward()
#             discriminator_fake_loss.backward()
#             # discriminator_loss.backward(retain_graph=True)
#             discriminator_optim.step()
#             discriminator_losses += [discriminator_loss.item()]
#
#             generator_optim.zero_grad()
#             fake_image_and_labels = torch.cat((fake, image_one_hot_labels), 1)
#             discriminator_fake_pred = discriminator(fake_image_and_labels)
#             # print(discriminator_fake_pred.shape)
#             generator_loss = criterion(discriminator_fake_pred, torch.ones_like(discriminator_fake_pred))
#             generator_loss.backward()
#             generator_optim.step()
#             generator_losses += [generator_loss.item()]
#
#             if epoch % display_epoch == 0 and epoch > 0:
#                 generator_mean = sum(generator_losses[-display_epoch:]) / display_epoch
#                 discriminator_mean = sum(discriminator_losses[-display_epoch:]) / display_epoch
#
#                 plot_image_from_tensor(fake)
#                 plot_image_from_tensor(real_x)
#                 dict_save = {
#                     'generator': generator.state_dict(),
#                     'generator-optim': generator_optim.state_dict(),
#                     'discriminator': discriminator.state_dict(),
#                     'discriminator-optim': discriminator_optim.state_dict(),
#                     'cfg': cfg
#                 }
#                 torch.save(dict_save, 'Model.pt')
#                 print(f'Step {epoch} : GeneratorLoss = {generator_mean} | DiscriminatorLoss = {discriminator_mean}')
#
#             print(
#                 f'\033[1;36m\r Current Index : {index} | Epoch : {epoch + 1}/{epochs} | GenLoss : {generator_loss.item()} | DiscLoss : {discriminator_loss.item()}',
#                 end='')
#
#         print()
#
#
# def train_n(epochs: int = 500, generator_model: torch.nn.Module = None, discriminator_model: torch.nn.Module = None,
#             generator_optim: torch.optim.Adam = None,
#             discriminator_optim: torch.optim.Adam = None,
#             use_init_weight: bool = True, lr: float = 2e-4, image_size: int = 28,
#             data_shape: typing.Union[tuple, list] = None, cfg=None,
#             display_epoch: int = 50, criterion: torch.nn = None,
#             device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu', z_dim: int = 120, batch_size: int = 128,
#             dataloader=None):
#     generator_model = generator_model.apply(weight_init)
#     discriminator_model = discriminator_model.apply(weight_init)
#     d_losses = []
#     img_list = []
#     g_losses = []
#     fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
#     real_label = 1
#     fake_label = 0
#     for epoch in range(epochs):
#         for index, (real, _) in enumerate(dataloader):
#             discriminator_model.zero_grad()
#             real = real.to(device)
#             current_batch = real.size(0)
#
#             label = torch.full((current_batch,), real_label, device=device, dtype=torch.float)
#
#             outputs = discriminator_model(real).view(-1)
#             error_discriminator_real = criterion(outputs, label)
#             error_discriminator_real.backward()
#
#             label.fill_(fake_label)
#
#             noise = create_noise(current_batch, z_dim)
#             fake = generator_model(noise)
#             # print(fake.shape)
#             outputs = discriminator_model(fake.detach()).view(-1)
#             error_discriminator_fake = criterion(outputs, label)
#             error_discriminator_fake.backward()
#
#             error_discriminator = error_discriminator_real + error_discriminator_fake
#
#             discriminator_optim.step()
#             label.fill_(real_label)
#             generator_model.zero_grad()
#
#             f = discriminator_model(fake).view(-1)
#             error_generator = criterion(f, label)
#             error_generator.backward()
#             generator_optim.step()
#
#             g_losses.append(error_generator.item())
#             d_losses.append(error_discriminator.item())
#
#             if epoch % display_epoch == 0 and epoch > 0:
#                 plot_image_from_tensor(fake)
#                 plot_image_from_tensor(real)
#                 dict_save = {
#                     'generator': generator_model.state_dict(),
#                     'generator-optim': generator_optim.state_dict(),
#                     'discriminator': discriminator_model.state_dict(),
#                     'discriminator-optim': discriminator_optim.state_dict(),
#                     'cfg': cfg
#                 }
#                 torch.save(dict_save, 'n_Model.pt')
#
#             print(
#                 f'\033[1;36m\r Current Index : {index} | Epoch : {epoch + 1}/{epochs} | GenLoss : {error_generator.item()} | DiscLoss : {error_discriminator.item()}',
#                 end='')
#
#         print()

if __name__ == "__main__":
    num_classes = 5
    init_fun, conv_net = stax.serial(
        Conv(32, (5, 5), (2, 2), padding="SAME"),
        BatchNorm(), Relu,
        Conv(32, (5, 5), (2, 2), padding="SAME"),
        BatchNorm(), Relu,
        Conv(10, (3, 3), (2, 2), padding="SAME"),
        BatchNorm(), Relu,
        Conv(10, (3, 3), (2, 2), padding="SAME"), Relu,
        Flatten,
        Dense(num_classes),
        LogSoftmax)
    print(conv_net)
