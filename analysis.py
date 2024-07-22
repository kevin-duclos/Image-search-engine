# Find and plot similar images
# Based on https://github.com/MathMagicx/JupyterNotebooks/blob/master/ImageRecommenderResnet18/Recommending%20Similar%20Images.ipynb
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.testing import assert_almost_equal
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')

# %% Get data
root = "/Users/kevinadmin/Desktop/PlanktoScope Processing/Test/export_12581_20240719_1809/LUMCON Oyster Larvae Sampling 2024-04-25_1/"
all_vecs = np.load(f"{root}/all_vecs.npy")
all_names = np.load(f"{root}/all_names.npy")

# Store as dictionary
vectors = {}
for i, vec in enumerate(all_vecs):
    vectors[all_names[i]] = vec

# %% Generate similarity matrix (slow for large datasets)
def getSimilarityMatrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns=keys, index=keys)
    return matrix

similarity_matrix = getSimilarityMatrix(vectors)

# %% Save similarity matrix to file
np.save(f"{root}/similarity_matrix.npy", similarity_matrix)

# %%
k = 20
similar_names = pd.DataFrame(index = similarity_matrix.index, columns = range(k))
similar_values = pd.DataFrame(index = similarity_matrix.index, columns = range(k))
for j in tqdm(range(similarity_matrix.shape[0])):
    kSimilar = similarity_matrix.iloc[j, :].sort_values(ascending = False).head(k)
    similar_names.iloc[j, :] = list(kSimilar.index)
    similar_values.iloc[j, :] = kSimilar.values


# %% Plot similar images
input_images = ['2024-04-25_21-22-41-731183_8.jpg', '2024-04-25_21-27-57-762984_156.jpg', '2024-04-25_21-25-46-347697_124.jpg', '2024-04-25_21-26-56-697196_120.jpg']


def getSimilarImages(image, simNames, simVals):
    if image in set(simNames.index):
        imgs = list(simNames.loc[image, :])
        vals = list(simVals.loc[image, :])
        if image in imgs:
            assert_almost_equal(max(vals), 1, decimal=5)
            imgs.remove(image)
            vals.remove(max(vals))
        return imgs, vals
    else:
        print("'{}' Unknown image".format(image))


def plotSimilarImages(image, similar_names, similar_values):
    simImages, simValues = getSimilarImages(image, similar_names, similar_values)
    fig = plt.figure(figsize=(10, 20))

    # now plot the  most simliar images
    for j in range(0, numCol * numRow):
        ax = []
        if j == 0:
            img = Image.open(os.path.join(root, image))
            ax = fig.add_subplot(numRow, numCol, 1)
            setAxes(ax, image, query=True)
        else:
            img = Image.open(os.path.join(root, simImages[j - 1]))
            ax.append(fig.add_subplot(numRow, numCol, j + 1))
            setAxes(ax[-1], simImages[j - 1], value=simValues[j - 1])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()

    plt.show()


for image in input_images:
    plotSimilarImages(image, similar_names, similar_values)


# %% Find similar images for one
from scipy.spatial.distance import cdist
numCol = 5
numRow = 5


def setAxes(ax, image, query=False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Query Image\n{0}".format(image), fontsize=12)
    else:
        ax.set_xlabel("Distance {1:1.3f}\n{0}".format(image, value), fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


image_name = '2024-04-25_21-22-41-731183_8.jpg'
idx = int(np.argwhere(all_names == image_name).squeeze())
target_vec = all_vecs[idx]
similarity = cdist(target_vec[None, ...], all_vecs, metric='cosine').squeeze()
similar_vecs = similarity.argsort()

similar_names = all_names[similar_vecs]
similar_values = similarity[similar_vecs]

fig = plt.figure(figsize=(10, 20))
for j in range(0, numCol * numRow):
    ax = []
    if j == 0:
        img = Image.open(os.path.join(root, image_name))
        ax = fig.add_subplot(numRow, numCol, 1)
        setAxes(ax, image_name, query=True)
    else:
        img = Image.open(os.path.join(root, similar_names[j - 1]))
        ax.append(fig.add_subplot(numRow, numCol, j + 1))
        setAxes(ax[-1], similar_names[j - 1], value=similar_values[j - 1])
    img = img.convert('RGB')
    plt.imshow(img)
    img.close()

plt.show()


# %% Plot similarities
fig, ax = plt.subplots()
ax.hist(similarity, bins=np.arange(0, 1, 0.02))
plt.show()

# %%
j = 10
fig, ax = plt.subplots()
img = Image.open(os.path.join(root, similar_names[j - 1]))
setAxes(ax, similar_names[j - 1], value=similar_values[j - 1])
img = img.convert('RGB')
plt.imshow(img)
img.close()
plt.show()
