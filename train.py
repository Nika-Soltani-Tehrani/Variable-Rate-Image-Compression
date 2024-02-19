import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import models
import config

# Define the model types
ONEWAY_MODELS = ["fc", "conv", "lstm"]
RESIDUAL_MODELS = ["fc_res", "conv_res", "lstm_res"]
MIX_MODELS = ["lstm_mix"]

# Create the model directory if it does not exist
if not os.path.exists(config.MODEL_PATH):
    os.makedirs(config.MODEL_PATH)

# Normalize function so given an image of range [0, 1] transforms it into a Tensor range [-1. 1]
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR LOADER
train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)


# ==============================================
# - CUSTOM FUNCTIONS
# ==============================================

def convert_second_to_minute(second):
    """Convert seconds to minutes and seconds."""
    minute = math.floor(second / 60)
    second -= minute * 60
    return "%dm %ds" % (minute, second)


def time_since(since, percent):
    """Calculate the elapsed and remaining time."""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%s (- %s)" % (convert_second_to_minute(s), convert_second_to_minute(rs))


def to_patches(x, patch_size):
    """Transform an image into patches of the given size."""
    num_patches_x = 32 // patch_size
    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            patch = x[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            patches.append(patch.contiguous())
    return patches


def train_oneway_model(model, patch, optimizer, criterion):
    """Train a one-way model on a single patch."""
    # Transform the tensor into Variable
    v_patch = Variable(patch)

    # Set gradients to Zero
    optimizer.zero_grad()

    # Forward + Backward + Optimize
    reconstructed_patches = model(v_patch)
    loss = criterion(reconstructed_patches, v_patch)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_residual_model(model, patch, optimizer, criterion):
    """Train a residual model on a single patch."""
    # Transform the tensor into Variable
    v_patch = Variable(patch)
    target_tensor = Variable(torch.zeros(v_patch.size()), requires_grad=False)
    losses = []

    # Set gradients to Zero
    optimizer.zero_grad()

    for p in range(config.REPEAT):
        # Forward + Backward + Optimize
        reconstructed_patches = model(v_patch, p)
        losses.append(criterion(reconstructed_patches, target_tensor))

        v_patch = reconstructed_patches
    loss = sum(losses)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_mix_model(model, patch, optimizer, criterion):
    """Train a mix model on a single patch."""
    # Transform the tensor into Variable
    v_patch = Variable(patch)
    losses = []

    # Set gradients to Zero
    optimizer.zero_grad()

    reconstructed_patches = model(v_patch)
    current_loss = criterion(reconstructed_patches, v_patch)
    losses.append(current_loss)

    loss = sum(losses)
    loss.backward()
    optimizer.step()

    return loss.data[0]


def train_model(model, patches, optimizer, criterion):
    """Train the model on a batch of patches."""
    running_loss = 0.0

    if config.MODEL_TYPE in ONEWAY_MODELS:
        for patch in patches:
            running_loss += train_oneway_model(model, patch, optimizer, criterion)

    elif config.MODEL_TYPE in RESIDUAL_MODELS:
        for patch in patches:
            running_loss += train_residual_model(model, patch, optimizer, criterion)

    else:
        model.reset_state()
        for patch in patches:
            running_loss += train_mix_model(model, patch, optimizer, criterion)

    return running_loss


def save_model(model, epoch, i):
    """Save the model to the model path."""
    torch.save(
        model.state_dict(),
        os.path.join(
            config.MODEL_PATH,
            config.MODEL_TYPE + "-p%d_b%d-%d_%d.pkl" % (config.PATCH_SIZE, config.CODED_SIZE, epoch + 1, i + 1)
        )
    )


def main():
    """Main function to train the model."""
    # Load the model
    model = models.setup()

    # Define the LOSS and the OPTIMIZER
    criterion = nn.MSELoss()
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    num_steps = len(train_loader)
    start = time.time()
    total_losses = []

    for epoch in range(config.NUM_EPOCHS):

        current_losses = []
        for i, data in enumerate(train_loader, 0):
            # Get the images
            images = data[0]

            # Transform into patches
            patches = to_patches(images, config.PATCH_SIZE)

            # Train the model on the patches
            running_loss = train_model(model, patches, optimizer, criterion)

            # STATISTICS:
            if (i + 1) % config.LOG_STEP == 0:
                print("(%s) [%d, %5d] loss: %.3f" %
                      (time_since(start, ((epoch * num_steps + i + 1.0) / (config.NUM_EPOCHS * num_steps))),
                       epoch + 1, i + 1, running_loss / config.LOG_STEP / config.NUM_PATCHES))
                current_losses.append(running_loss / config.LOG_STEP / config.NUM_PATCHES)
                running_loss = 0.0

            # SAVE:
            if (i + 1) % config.SAVE_STEP == 0:
                save_model(model, epoch, i)

        total_losses.append(current_losses)
        save_model(model, epoch, i)

    print("__TRAINING DONE=================================================")


if __name__ == '__main__':
    main()
