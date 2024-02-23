import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import config
import models

# Create the output directory if it does not exist
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

# Normalize the input images
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Load the CIFAR100 test set
test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

# Get an iterator over the test set
dataiter = iter(test_loader)


# ==============================================
# - CUSTOM FUNCTIONS
# ==============================================

def img_save(img, name):
    """Save an image to the output path."""
    img = img / 2 + 0.5  # denormalize
    saving_path = os.path.join(config.OUTPUT_PATH, name + ".png")
    torchvision.utils.save_image(img, saving_path)


def to_patches(x, patch_size):
    """Transform an image into patches of the given size."""
    num_patches_x = 32 // patch_size
    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            patch = x[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            patches.append(patch.contiguous())
    return patches


def reconstruct_patches(patches):
    """Transform the patches into the image."""
    batch_size = patches[0].size(0)
    patch_size = patches[0].size(2)
    num_patches_x = 32 // patch_size
    reconstructed = torch.zeros(batch_size, 3, 32, 32)
    p = 0
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            reconstructed[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches[
                p].data
            p += 1
    return reconstructed


def eval_model(model, imgs):
    """Evaluate the model on a batch of images."""
    # Patch the image
    patches = to_patches(imgs, config.PATCH_SIZE)
    reconstructed_patches = []
    if config.RESIDUAL is None:
        model.reset_state()
    for p in patches:
        if config.RESIDUAL:
            outputs = model.sample(Variable(p))
        else:
            outputs = model(Variable(p))
        reconstructed_patches.append(outputs)
    # Transform the patches into the image
    outputs = reconstruct_patches(reconstructed_patches)
    return outputs


def main():
    """Main function to evaluate the model."""
    print(":::::::::::::::::Starting eval:::::::::::::::::")
    for i in range(config.NUM_SAMPLES // config.BATCH_SIZE):
        imgs, _ = next(dataiter)
        print(imgs.shape)
        img_save(torchvision.utils.make_grid(imgs), "prova_" + str(i))

        # Evaluate the model on the images
        outputs = eval_model(model, imgs)
        img_save(torchvision.utils.make_grid(outputs), "prova_" + str(i) + "_decoded")


# =============================================================================
# - PARAMETERS
# =============================================================================

if __name__ == "__main__":
    # Load the model
    model = models.setup()

    # Load the saved model
    path_to_model = os.path.join(config.MODEL_PATH,
                                 config.MODEL_TYPE + "-p%d_b%d-%d_%d.pkl" % (config.PATCH_SIZE,
                                                                             config.CODED_SIZE,
                                                                             config.LOAD_EPOCH,
                                                                             config.LOAD_ITER)
                                 )
    path_to_model = '/home/nika/private/CE Project/saved_models/fc_res-p8_b4-1_250.pkl'
    model.load_state_dict(torch.load(path_to_model))
    main()
