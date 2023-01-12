import torch

use_gpu = True if torch.cuda.is_available() else False

# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=use_gpu)
# this model outputs 256 x 256 pixel images
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-256',
#                        pretrained=True, useGPU=use_gpu)

if __name__ == "__main__":
    # print(help(model))
    model.train()

    # num_images = 4
    # noise, _ = model.buildNoiseData(num_images)
    # with torch.no_grad():
    #     generated_images = model.test(noise)
    #
    # # let's plot these images using torchvision and matplotlib
    # import matplotlib.pyplot as plt
    # import torchvision
    #
    # grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    # plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    # plt.show()