# StableGAN

StableGAN is a Text to image Model you can give the prompt that you want to see the image of it and you will get the
image of that, and you also can specify which version you want to see your image in

my goal is an AI free for everybody and open-source
and I provide this model usage at

- Web Application
- Desktop User Interface
- Engine and Command line (CLI)

# About StableGAN 👋

Hi there, my name is StableGAN and I am an open-source AI designed to make machine learning accessible and free for
everybody. My ultimate goal is to provide a powerful, yet easy-to-use platform that enables people from all walks of
life to harness the power of artificial intelligence and machine learning in creating and developing innovative
solutions.

At the core of my design is the mission to enable anyone to use machine learning to solve complex problems by providing
an array of features in a user-friendly manner. I am compatible with desktop UI, and web applications, and can be utilized
through a command line engine. My versatility is what makes me stand out from the rest and sets me apart as a great
option for anyone looking to create practical and optimized solutions for their business, research, or personal use.

My development team believes in the democratization of technology and knowledge, hence why we have made StableGAN
free to use for everyone. We strongly believe that nobody should be left behind when it comes to this cutting-edge
technology and our open-source platform is our contribution to achieving that goal.

If you are looking for an AI that values your time and resources, with powerful tools that can help you build
state-of-the-art solutions using stable diffusion, then StableGAN is the perfect AI for you.

# StableGAN in HuggingfaceDiffusers

Here's how to import and use the model

```python
from diffusers import StableDiffusionPipeline

pipe_line = StableDiffusionPipeline.from_pretrained('erfanzar/StableGAN')
```

# Requierments 

### installing requirements

``` shell
pip install datasets sentencepiece accelerate wandb peft tensorboard transformers gradio diffusers erutils safetensors -q --upgrade
```

## Simple and EasyUse
Ui and auto load support some other models instead of StableGAN such as `stabilityai/stable-diffusion-2-1`

``` shell
git clone https://github.com/erfanzar/StableGAN.git
cd StableGAN
git pull
python app.py --model_path='stabilityai/stable-diffusion-2-1'
```

or using StableGAN
```shell
python app.py --model_path='erfanzar/StableGAN'
```

# Desktop UI

in the new version, I created a user interface that you can simply use a model with

this UI is built using Python [felt](https://flet.dev/)

```shell
python3 desktop_app.py
```

## Project Goal

The primary objective of StableGAN is to provide an AI solution that is accessible to everyone and completely open source. The model can be utilized through various interfaces, including:

1. pre-training some cool GenetativeModel for custom use cases
2. developing and researching Generative Models in Art and multimodal Field
3. creating an easy-to-use and user-friendly UI to make it using the model and evaluation process easier
4. Engine and Command Line Interface (CLI)
5. Web Application
6. Desktop User Interface  

## HuggingFace and Diffusers

[Hugging Face](https://huggingface.co/) is an open-source software library created to make Natural Language Processing (
NLP) accessible to
developers. It provides a variety of pre-trained models and tools for developers to build state-of-the-art NLP systems.
Hugging Face offers easy-to-use APIs that allow developers to integrate its models with their applications quickly and
efficiently.

I used HuggingFace for a part of the model (`CLIP`)
and for the sake of better training, we edited a part of the models that we got from HuggingFace (`CLIP Text Model`)

to make it faster cause there was some implementation that wasn't required and a part of the algorithm was created by myself
and hugging face does not include those algorithms
