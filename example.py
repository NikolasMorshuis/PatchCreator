import numpy as np
from patchcreator import *

#input = np.zeros((1, 1, 100, 100, 20))
#patch_shape = np.array([1, 1, 12, 12, 12])
input = np.zeros((100, 100))
patch_shape = np.array([12, 12])
indices = index_creator(input, patch_shape, 2)
print(indices)


def predict_patches(input, patch_shape, indices):
    """
    :param input: Numpy matrix that you like to get the patches from
    :param patch_shape: The size that the patches should have
    :param indices: The start indices of every patch of every dimension, as it is returned from PatchCreator
    :return: A numpy matrix of size (n_patches, patch_shape) that lists the predicted patches.
    """
    # Basic framework for the prediction of the patches. Will need to be adjusted by the user in order to work properly
    predicted_patches = np.zeros(patch_shape)
    predicted_patches = np.expand_dims(predicted_patches, 0)
    predicted_patches = np.repeat(predicted_patches, len(indices), 0)
    for i in range(len(indices)):
        """" How you predict the patches depends on which framework you use, which dimensionality your data has etc.
         An example is provided below on how the prediction process can look like when using pytorch. """
        # patch = input[indices[0]:indices[0]+patch_shape[0], indices[1]:indices[1]+patch_shape[1]]
        # patch_torch = torch.tensor(patch)
        # prediction = model(patch_torch)
        # prediction_numpy = prediction.numpy()
        # predicted_patches[i] = prediction_numpy

        # for now (comment when adjusted to your program):
        predicted_patches[i] = np.ones(patch_shape)
        pass
    return predicted_patches

# for now:
predicted_patches =  predict_patches(input, patch_shape, indices)

matrix = aggregate_patches(np.shape(input), indices, predicted_patches)
