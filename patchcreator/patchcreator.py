import numpy as np
from itertools import product
import warnings
from scipy import stats


def index_creator(input, patch_shape, overlap=None):
    """
    :param input: Numpy matrix that you like to get the patches from
    :param patch_shape: The size that the patches should have
    :param overlap: The minimum overlap between the patches
    :return: indices to indicate the start point of each patch for each dimension
    """
    shape = np.shape(input)
    print(shape)
    dims = len(shape)
    if isinstance(overlap, int):
        overlap = np.ones_like(shape) * overlap
        overlap = np.where(np.array(shape) == 1., 0, overlap)
    if isinstance(overlap, list):
        overlap = np.array(overlap)
    assert (len(shape) == len(patch_shape)), 'The input shape and the patch size should ' \
                                            'have the same number of dimensions'
    if (shape < patch_shape).any():
        warnings.warn('The patch size should not be larger than the shape of the input in any dimension, '
                      'We just make the patch size smaller for now.', UserWarning, stacklevel=2)
        patch_shape = np.minimum(shape, patch_shape)

    # Calculate minimum number of patches for each dimension:
    if overlap is not None:
        patches_per_dimension = np.ceil((overlap - shape) / (overlap - patch_shape)).astype(int)
        min_overlap = overlap
    else:
        patches_per_dimension = np.ceil(shape / patch_shape).astype(int)
        # Calculate the overlap that the patches will have:
        total_overlap = patches_per_dimension * patch_shape - shape
        min_overlap = np.where(patches_per_dimension > 1, np.floor(total_overlap / (patches_per_dimension - 0.999)),
                               patch_shape)

    # Distribute the patches per dimension equally over the dimensions:
    dim_counter = 0
    n_matrices = np.int(np.prod(patches_per_dimension))
    indices = []
    indices_matrix = np.zeros((n_matrices, dims))
    while dim_counter < dims:
        if patches_per_dimension[dim_counter] <= 1:
            indices_dim = np.array([0])
        else:
            indices_dim = np.zeros((patches_per_dimension[dim_counter]))
            indices_dim[0] = 0
            if len(indices_dim) > 1:
                indices_dim[-1] = shape[dim_counter] - patch_shape[dim_counter]
            if len(indices_dim) > 2:
                new_dist = np.round((indices_dim[-1] - indices_dim[0]) / (len(indices_dim) - 1))
                for i in range(len(indices_dim) - 2):
                    indices_dim[i + 1] = (i + 1) * new_dist
        indices.append(indices_dim)
        dim_counter += 1
    for c, x in enumerate(product(*indices)):
        indices_matrix[c] = x
    return np.ndarray.astype(indices_matrix, np.int)


def predict_patches(input, patch_shape, indices):
    """
    :param input: Numpy matrix that you like to get the patches from
    :param patch_shape: The size that the patches should have
    :param indices: The start indices of every patch of every dimension, as it is returned from PatchCreator
    :return: A numpy matrix of size (n_patches, patch_shape) that lists the predicted patches.
    """
    # Basic framework for the prediction of the patches. Will need to be adjusted by the user in order to work properly
    predicted_patches = np.zeros_like(patch_shape)
    predicted_patches = np.expand_dims(predicted_patches, 0)
    predicted_patches = np.repeat(predicted_patches, len(indices), axis=0)
    for i in range(len(indices)):
        """" How you predict the patches depends on which framework you use, which dimensionality your data has etc.
         An example is provided below on how the prediction process can look like when using pytorch. """
        # patch = input[indices[0]:indices[0]+patch_shape[0], ...]
        # patch_torch = torch.tensor(patch)
        # prediction = model(patch_torch)
        # prediction_numpy = prediction.numpy()
        # predicted_patches[i] = prediction_numpy
        pass
    return(predicted_patches)


def create_prob_matrix(shape_matrix, x, i):
    c = 0
    while c < i:
        x = np.expand_dims(x, c)
        x = np.repeat(x, shape_matrix[c], c)
        c += 1
    c = i+1
    while c < len(shape_matrix):
        x = np.expand_dims(x, c)
        x = np.repeat(x, shape_matrix[c], c)
        c += 1
    return(x)


def aggregate_patches(output_shape, indices, patches, cov=0.01):
    """
    :param output_shape: The final shape that the output will have
    :param indices: The matrix of indices as returned by PatchCreator
    :param patches: The list of patches as returned by predict_patches. Shape: (len(indices), patch_shape).
    :param cov: Define the sigma used. If you encounter artifacts at the patch-borders, you might want to reduce the
                value
    :return: The predicted output matrix
    """
    dim_patches = len(indices[0])
    print('Dimensionality of the patches: {}'.format(dim_patches))
    assert (dim_patches <= 5), 'The dimensionality of the patches should be smaller or equal to 5.' \
                               ' You can however add the process for any dimensionality by adding the code' \
                               ' for the new dimensionality below'
    # The patches are combined using gaussian functions
    pure_weight_matrix = np.zeros(output_shape)
    weighted_matrix = np.zeros(output_shape)
    weight_matrix = np.zeros(output_shape)
    output_matrix = np.zeros(output_shape)
    patch_shape = np.shape(patches)[1:]
    # Create the probability matrix:
    matrix = np.ones_like(patches[0])
    for i in range(len(patch_shape)):
        if patch_shape[i] > 1:
            x = np.linspace(0, 1, patch_shape[i])
            x = stats.multivariate_normal.pdf(x, mean=0.5, cov=cov)
            weight_1d = create_prob_matrix(np.shape(matrix), x, i)
            matrix *= weight_1d
    for i in range(len(indices)):
        if dim_patches == 1:
            pure_weight_matrix[indices[i][0]:indices[i][0]+patch_shape[0]] += matrix
            weighted_matrix[indices[i][0]:indices[i][0]+patch_shape[0]] += matrix * patches[i]
        if dim_patches == 2:
            pure_weight_matrix[indices[i][0]:indices[i][0]+patch_shape[0], indices[i][1]:indices[i][1]+patch_shape[1]] \
                += matrix
            weighted_matrix[indices[i][0]:indices[i][0]+patch_shape[0], indices[i][1]:indices[i][1]+patch_shape[1]] \
                += matrix * patches[i]
        elif dim_patches == 3:
            pure_weight_matrix[indices[i][0]:indices[i][0]+patch_shape[0], indices[i][1]:indices[i][1]+patch_shape[1],
            indices[i][2]:indices[i][2]+patch_shape[2]] += matrix
            weighted_matrix[indices[i][0]:indices[i][0]+patch_shape[0], indices[i][1]:indices[i][1]+patch_shape[1],
            indices[i][2]:indices[i][2]+patch_shape[2]] += matrix * patches[i]
        elif dim_patches == 4:
            pure_weight_matrix[indices[i][0]:indices[i][0]+patch_shape[0], indices[i][1]:indices[i][1]+patch_shape[1],
            indices[i][2]:indices[i][2]+patch_shape[2], indices[i][3]:indices[i][3]+patch_shape[3]] += matrix
            weighted_matrix[indices[i][0]:indices[i][0]+patch_shape[0], indices[i][1]:indices[i][1]+patch_shape[1],
            indices[i][2]:indices[i][2]+patch_shape[2], indices[i][3]:indices[i][3]+patch_shape[3]] += matrix * patches[i]
        elif dim_patches == 5:
            pure_weight_matrix[indices[i][0]:indices[i][0] + patch_shape[0],
            indices[i][1]:indices[i][1] + patch_shape[1],
            indices[i][2]:indices[i][2] + patch_shape[2], indices[i][3]:indices[i][3] + patch_shape[3],
            indices[i][4]:indices[i][4] + patch_shape[4]] += matrix
            weighted_matrix[indices[i][0]:indices[i][0] + patch_shape[0], indices[i][1]:indices[i][1] + patch_shape[1],
            indices[i][2]:indices[i][2] + patch_shape[2], indices[i][3]:indices[i][3] + patch_shape[3],
            indices[i][4]:indices[i][4]+patch_shape[4]] += matrix * patches[i]
        output_matrix = (weight_matrix * output_matrix + weighted_matrix) / np.where(
            weight_matrix + pure_weight_matrix == 0, 1, weight_matrix + pure_weight_matrix)
        weight_matrix += pure_weight_matrix
    return(output_matrix)
