import numpy as np
import patchcreator

# Input matrix, usually the input to the model:
input = np.zeros((100, 100))

# Shape of the patches:
patch_shape = np.array([12, 12])

"""
You just need the input, the patch-shape, and the minimum overlap between the patches to create the indices.
"""
indices = patchcreator.index_creator(input=input, patch_shape=patch_shape, overlap=2)
print(indices)

"""
The indices can be used to predict the patches. Note that you should replace this function by your own.
The implementation depends on which architecture you use.
"""
# for now:
predicted_patches = patchcreator.predict_patches(input, patch_shape, indices)


"""
The results can be combined by the patchcreator. Note that this function weights the boarder of each patch
 less strongly than the middle of each patch, because the border can have border artifacts.
"""
matrix = patchcreator.aggregate_patches(output_shape=np.shape(input), indices=indices, patches=predicted_patches)
