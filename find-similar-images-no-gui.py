import numpy as np
import pandas as pd
from os import path
from PIL import Image
from glob import glob
from pyecotaxa.archive import read_tsv, write_tsv
from scipy.spatial.distance import cdist
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')
plt.style.use('seaborn-v0_8-talk')


# %% Get data
def read_data(root):
    image_vecs = np.load(f"{root}/data/all_vecs.npy")
    image_names = np.load(f"{root}/data/all_names.npy")
    image_names = np.array([image_name.replace('.jpg', '') for image_name in image_names])
    return image_vecs, image_names

# search_image_path = "/Users/kevinadmin/Desktop/Image Similarity/LUMCON Oyster Larvae Sampling 2024-05-02_1"
search_image_path = "/Users/kevinadmin/Desktop/Image Similarity/LUMCON Oyster Larvae Sampling 2024-04-25_1"
target_image_path = "/Users/kevinadmin/Desktop/Image Similarity/Oyster Larvae Training Set"

# Get data files
try:
    search_image_vecs, search_image_names = read_data(search_image_path)
    target_image_vecs, target_image_names = read_data(target_image_path)
except Exception as e:
    print(e)


# %% Get similar images for one target image
def get_similar_images(vecs, names, target_vec, n_images):
    distances = cdist(target_vec[None, ...], vecs, metric='cosine').squeeze()
    top_image_indices = distances.argsort()[range(n_images)]
    top_image_names = names[top_image_indices]
    top_image_names = np.array([image_name.replace('.jpg', '') for image_name in top_image_names])
    top_image_distances = distances[top_image_indices]
    top_images = pd.DataFrame(data={'index': top_image_indices, 'name': top_image_names, 'distance': top_image_distances})
    return top_images


# %% Get similar images for all target images
def get_all_similar_images(target_image_vecs, n_images):
    all_top_images = []
    for target_image_vec in tqdm(target_image_vecs):
        top_images = get_similar_images(search_image_vecs, search_image_names, target_image_vec, n_images)
        all_top_images.append(top_images)

    all_top_images = pd.concat(all_top_images)

    # Remove duplicates
    all_top_images.drop_duplicates(subset=['name'], inplace=True)

    # Remove target images
    all_top_images = all_top_images[~all_top_images['name'].isin(target_image_names)]

    # Sort by distance
    all_top_images.sort_values(by='distance', inplace=True)
    all_top_images.reset_index(inplace=True)

    return all_top_images

# %% Get matches for one target image
target_image_vec = target_image_vecs[0]
image_matches = get_similar_images(search_image_vecs, search_image_names, target_image_vec, 30)

# %% Get matches for all target images
all_image_matches = get_all_similar_images(target_image_vecs, 10)

# %% Show images
n_cols = 10
n_rows = 10

images = all_image_matches
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12))
for i, ax in enumerate(axes.flat):
    try:
        # image_name = all_image_matches.iloc[i]['name']
        # image_name = images.iloc[i]['name']
        image_name = images.iloc[np.random.randint(images.shape[0])]['name']
        image = Image.open(path.join(search_image_path, image_name + '.jpg'))
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # ax.set_title(image_name)
    except:
        ax.axis('off')

plt.tight_layout()
