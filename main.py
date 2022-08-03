import math
import random
import time
import torch
from torch.optim import Adam
from torch.utils.data import Dataset as DS, DataLoader
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from torch import nn
import torchvision
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from tutorial import PatchDiscriminator, init_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CHANNELS = 3


def time_since(since: float) -> str:
    """
    Returns the time since an instant
    :param since: The start time
    :return: a string with minutes and seconds
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class ImageDataset(DS):
    """
    Manages supplying random pairs of art, non_art images
    """
    def __init__(self) -> None:
        self.art_path = "path/to/portraits/dataset"
        self.art_len = len(os.listdir(self.art_path))
        self.art_images = os.listdir(self.art_path)

        self.non_art_path = "path/to/humans/dataset"
        self.non_art_len = len(os.listdir(self.non_art_path))
        self.images = os.listdir(self.non_art_path)

    def __len__(self) -> int:
        """
        :return: The number of images per epoch -- doesn't affect training times
        """
        return 100

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Supplies a random art and non art image, as tensors
        :param idx: Index -- **this is ignored**
        :return: (art, non-art)
        """
        idx = random.randint(0, self.art_len) - 1
        art = Image.open(self.art_path + self.art_images[idx]).resize((256, 256)).convert("RGB")
        art = np.array(art).astype("float32")
        art = torchvision.transforms.ToTensor()(art) / 256

        idx = random.randint(0, self.non_art_len) - 1
        non_art = Image.open(self.non_art_path + self.images[idx]).resize((256, 256)).convert("RGB")
        non_art = np.array(non_art).astype("float32")
        non_art = torchvision.transforms.ToTensor()(non_art) / 256

        return art.to(device), non_art.to(device)


def train_GAN(generator: nn.Module, data: DS, epochs: int = 1) -> None:
    """
    Runs the GAN training loop, updating the generator. Each epoch, the
    generator and discriminator state_dicts are saved to "generator.pt" and
    "discriminator.pt", respectively.
    :param data: Dataset w/ images
    :param generator: The generator model to be trained
    :param epochs: Number of training cycles
    :return: None
    """

    data = DataLoader(data, batch_size=8, shuffle=True)
    losses = []

    discriminator = PatchDiscriminator(input_c=NUM_CHANNELS).to(device)
    discriminator = init_weights(discriminator)
    discriminator.load_state_dict(torch.load("discriminator.pt"))
    discriminator.train()

    GANCriterion = nn.BCEWithLogitsLoss()
    L1criterion = nn.L1Loss()

    generator_optimizer = Adam(generator.parameters(), lr=1e-3)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=1e-3)

    for model in generator, discriminator:
        for param in model.parameters():
            param.requires_grad = True

    for epoch in range(epochs):
        for n, (art, non_art) in enumerate(data):

            generator_yhat = generator(non_art)

            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()

            discriminator_yhat = discriminator(generator_yhat.clone().detach())
            discriminator_y = discriminator(art)

            discriminator_yhat_loss = GANCriterion(discriminator_yhat, torch.tensor(0.0, device=device).expand_as(discriminator_yhat))
            discriminator_y_loss = GANCriterion(discriminator_y, torch.tensor(1.0, device=device).expand_as(discriminator_y))

            discriminator_loss = (discriminator_yhat_loss + discriminator_y_loss) * .5
            discriminator_loss.backward()
            discriminator_optimizer.step()

            discriminator_feedback = discriminator(generator_yhat)
            generator_GAN_loss = GANCriterion(discriminator_feedback, torch.tensor(1.0, device=device).expand_as(discriminator_feedback))
            generator_L1_loss = L1criterion(generator_yhat, non_art) * 85 # identity loss to incentivise staying close to input

            generator_loss = generator_GAN_loss + generator_L1_loss
            generator_loss.backward()
            generator_optimizer.step()

            losses.append(generator_L1_loss.detach().cpu().numpy())

        torch.save(generator.state_dict(), "generator.pt")
        torch.save(discriminator.state_dict(), "discriminator.pt")

    plt.plot(losses)
    plt.savefig("generator-loss-plot.png")

    torch.save(generator.state_dict(), "generator.pt")
    torch.save(discriminator.state_dict(), "discriminator.pt")


def create_generator(path: str = None) -> nn.Module:
    """
    factory for generator objects
    :param path: Path to state_dict. If none, defaults to pretrained resnet
    :return: The object
    """
    body = create_body(resnet18(), pretrained=True, n_in=NUM_CHANNELS, cut=-2)
    generator = DynamicUnet(body, NUM_CHANNELS, (256, 256))
    generator.to(device)
    if path is not None:
        generator.load_state_dict(torch.load(path))
    return generator


if __name__ == '__main__':
    training = False
    predicting = not training

    generator = create_generator("generator.pt")

    if training:
        data = ImageDataset()

        generator.train()
        train_GAN(generator, data, 1000)

    if predicting:
        generator.eval()

        path = "path/to/image"
        img = Image.open(path).resize((256, 256)).convert("RGB")
        arr = np.array(img).astype("float32") / 256
        img = torchvision.transforms.ToTensor()(arr)
        img = img.to(device)

        y_hat = generator(img.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        y_hat = np.stack([y_hat[0], y_hat[1], y_hat[2]], axis=2)
        y_hat = np.clip(y_hat, 0, 1)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(arr)
        ax[0].axis('off')
        ax[1].imshow(y_hat)
        ax[1].axis('off')
        plt.show()
