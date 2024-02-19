# Training Phase
MODEL_TYPE = 'fc_res'  # Name of the model to be used: fc, fc_rec, conv, conv_rec, lstm
RESIDUAL = True  # Set True if the model is residual, otherwise False
PATCH_SIZE = 4  # Size for the encoded subdivision of the input image
NUM_PATCHES = (32 // PATCH_SIZE) ** 2  # Number of patches per image
CODED_SIZE = 32  # Number of bits representing the encoded patch
REPEAT = 8  # Number of passes for recursive architectures
BATCH_SIZE = 4  # Mini-batch size
NUM_EPOCHS = 30  # Number of iterations where the system sees all the data
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0
MOMENTUM = 0.9
LOG_STEP = 10  # Step size for printing the log info
SAVE_STEP = 100  # Step size for saving the trained models
MODEL_PATH = '/home/nika/private/CE Project/saved_models2/'

# Evaluation Phase
OUTPUT_PATH = '/home/nika/private/CE Project/test_imgs/'  # Path where the output images should be saved
LOAD_ITER = 50  # Iteration which the model to be loaded was saved
LOAD_EPOCH = 1  # Epoch in which the model to be loaded was saved
NUM_SAMPLES = 20  # Number of pictures to be plotted
