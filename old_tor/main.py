import torch
import torch.nn as nn
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import random, torch, os, numpy as np
import torch.nn as nn
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import torch
import mat
plotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.03
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 1
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_P = "genp.pth.tar"
CHECKPOINT_GEN_M = "genm.pth.tar"
CHECKPOINT_DISC_P = "discp.pth.tar"
CHECKPOINT_DISC_M = "discm.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


# %%
# instance norm

# They mentioned in the paper that using padding_mode="reflect" helped to reduce artifacts
# The 1 is the padding
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x);


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1,
                                padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)


# %%

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2,
                          padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2,
                          padding=1),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3,
                          stride=2, padding=1, output_padding=1),
                ConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3,
                          stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features * 1, img_channels, kernel_size=7, stride=1,
                              padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


def test_gen():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)


# load data

class PictureMonetDataset(Dataset):
    def __init__(self, root_monet, root_picture, transform=None):
        self.root_monet = root_monet
        self.root_picture = root_picture
        self.transform = transform

        self.monet_images = os.listdir(root_monet)
        self.picture_images = os.listdir(root_picture)
        self.length_dataset = max(len(self.monet_images),
                                  len(self.picture_images))  # 1000, 1500
        self.monet_len = len(self.monet_images)
        self.picture_len = len(self.picture_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        monet_img = self.monet_images[index % self.monet_len]
        picture_img = self.picture_images[index % self.picture_len]

        monet_path = os.path.join(self.root_monet, monet_img)
        picture_path = os.path.join(self.root_picture, picture_img)

        monet_img = np.array(Image.open(monet_path).convert("RGB"))
        picture_img = np.array(Image.open(picture_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=monet_img, image0=picture_img)
            monet_img = augmentations["image"]
            picture_img = augmentations["image0"]

        return monet_img, picture_img


"""
UTILS
"""


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Show samples

def show_samples(dataset):
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(dataset.__getitem__(0)[0])
    axarr[0, 1].imshow(dataset.__getitem__(1)[0])
    axarr[1, 0].imshow(dataset.__getitem__(0)[1])
    axarr[1, 1].imshow(dataset.__getitem__(1)[1])


""" TRAINING """


def train_fn(disc_P, disc_M, gen_M, gen_P, loader, opt_disc, opt_gen, l1, mse, d_scaler,
             g_scaler):
    P_reals = 0
    P_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (monet, picture) in enumerate(loop):
        monet = monet.to(DEVICE)
        picture = picture.to(DEVICE)

        # Train Discriminators P and M
        with torch.cuda.amp.autocast():
            fake_picture = gen_P(monet)
            D_P_real = disc_P(picture)
            D_P_fake = disc_P(fake_picture.detach())
            P_reals += D_P_real.mean().item()
            P_fakes += D_P_fake.mean().item()
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            fake_monet = gen_M(picture)
            D_M_real = disc_M(monet)
            D_M_fake = disc_M(fake_monet.detach())
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            # put it togethor
            D_loss = (D_P_loss + D_M_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators P and M
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_P_fake = disc_P(fake_picture)
            D_M_fake = disc_M(fake_monet)
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))

            # cycle loss
            cycle_monet = gen_M(fake_picture)
            cycle_picture = gen_P(fake_monet)
            cycle_monet_loss = l1(monet, cycle_monet)
            cycle_picture_loss = l1(picture, cycle_picture)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_monet = gen_M(monet)
            identity_picture = gen_P(picture)
            identity_monet_loss = l1(monet, identity_monet)
            identity_picture_loss = l1(picture, identity_picture)

            # add all togethor
            G_loss = (
                    loss_G_M
                    + loss_G_P
                    + cycle_monet_loss * LAMBDA_CYCLE
                    + cycle_picture_loss * LAMBDA_CYCLE
                    + identity_picture_loss * LAMBDA_IDENTITY
                    + identity_monet_loss * LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_picture * 0.5 + 0.5,
                       f"/content/saved_images/photo_{idx}.png")
            save_image(fake_monet * 0.5 + 0.5, f"/content/saved_images/monet_{idx}.png")

        loop.set_postfix(P_real=P_reals / (idx + 1), P_fake=P_fakes / (idx + 1))
        return G_loss, D_loss



class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, 'outputs/best_model.pth')


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'outputs/final_model.pth.tar')


def save_plots(G_loss, D_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        G_loss, color='orange', linestyle='-',
        label='Generator Loss'
    )
    plt.plot(
        D_loss, color='red', linestyle='-',
        label='Discriminator Loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')




def main():
    disc_P = Discriminator(in_channels=3).to(DEVICE)
    disc_M = Discriminator(in_channels=3).to(DEVICE)
    gen_M = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    gen_P = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    opt_disc = optim.Adam(
        list(disc_P.parameters()) + list(disc_M.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_M.parameters()) + list(gen_P.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    save_best_model = SaveBestModel()
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN_P, gen_P, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_GEN_M, gen_M, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC_P, disc_P, opt_disc, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC_M, disc_M, opt_disc, LEARNING_RATE,
        )

    dataset = PictureMonetDataset(
        root_picture=TRAIN_DIR + "/photo", root_monet=TRAIN_DIR + "/monet",
        transform=transforms
    )
    val_dataset = PictureMonetDataset(
        root_picture=VAL_DIR + "/photo", root_monet=VAL_DIR + "/monet",
        transform=transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    generator_loss, discriminator_loss = [], []
    for epoch in range(NUM_EPOCHS):
        G_loss, D_loss = train_fn(disc_P, disc_M, gen_M, gen_P, loader, opt_disc,
                                  opt_gen, L1, mse, d_scaler, g_scaler)
        generator_loss.append(G_loss)
        discriminator_loss.append(D_loss)
        if SAVE_MODEL:
            save_checkpoint(gen_P, opt_gen, filename=CHECKPOINT_GEN_P)
            save_checkpoint(gen_M, opt_gen, filename=CHECKPOINT_GEN_M)
            save_checkpoint(disc_P, opt_disc, filename=CHECKPOINT_DISC_P)
            save_checkpoint(disc_M, opt_disc, filename=CHECKPOINT_DISC_M)

    # if SAVE_MODEL:
    #    save_model(NUM_EPOCHS, gen_P, opt_gen)
    #    save_model(NUM_EPOCHS, gen_M, opt_gen)
    #    save_model(NUM_EPOCHS, disc_P, opt_disc)
    #    save_model(NUM_EPOCHS, disc_M, opt_disc)
    # save_plots(generator_loss, discriminator_loss)

