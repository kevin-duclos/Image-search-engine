import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from glob import glob
from pyecotaxa.archive import read_tsv, write_tsv
from scipy.spatial.distance import cdist


# %%
@st.cache_data
def read_data(root, search_image_path):
    all_vecs = np.load(f"{root}/data/all_vecs.npy")
    all_names = np.load(f"{root}/data/all_names.npy")
    all_names = np.array([image_name.replace('.jpg', '') for image_name in all_names])
    # Get search images
    # search_image_data = read_tsv(search_image_path)
    # search_image_names = search_image_data['object_id'].to_numpy()
    # search_image_names = np.array([image_name.replace('.jpg', '') for image_name in search_image_names])
    search_image_vecs = np.load(f"{search_image_path}/data/all_vecs.npy")
    search_image_names = np.load(f"{search_image_path}/data/all_names.npy")
    search_image_names = np.array([image_name.replace('.jpg', '') for image_name in search_image_names])
    return all_vecs, all_names, search_image_vecs, search_image_names

st.title('Plankton picker')
# st.session_state["filepath_in"] = st.text_input("File path", "")
st.session_state["filepath_images"] = st.text_input("Images path", "/Users/kevinadmin/Desktop/Image Similarity/LUMCON Oyster Larvae Sampling 2024-05-02_1")
# st.session_state["filepath_search"] = st.text_input("EcoTaxa path", "/Users/kevinadmin/Desktop/Image Similarity/LUMCON Oyster Larvae Sampling 2024-05-02_1/data/ecotaxa_export_12759_20240726_1641_bivalve.tsv")
st.session_state["filepath_search"] = st.text_input("EcoTaxa path", "/Users/kevinadmin/Desktop/Image Similarity/Oyster Larvae Training Set")

# Get data files
try:
    all_vecs, all_names, search_image_vecs, search_image_names = read_data(st.session_state["filepath_images"], st.session_state["filepath_search"])
    st.subheader(st.session_state["filepath_images"])
    st.subheader(st.session_state["filepath_search"])
except Exception as e:
    st.warning(e)


# %%
def get_similar_images(vecs, names, target_vec, n_images):
    # idx = int(np.argwhere(all_names == search_image_name).squeeze())
    # target_vec = vecs[idx]
    distances = cdist(target_vec[None, ...], vecs, metric='cosine').squeeze()
    top_image_indices = distances.argsort()[range(n_images)]
    top_image_names = names[top_image_indices]
    top_image_names = np.array([image_name.replace('.jpg', '') for image_name in top_image_names])
    top_image_distances = distances[top_image_indices]
    top_images = pd.DataFrame(data={'index': top_image_indices, 'name': top_image_names, 'distance': top_image_distances})
    return top_images


# %% Get similar images for all search images
@st.cache_data
def get_all_similar_images(search_image_vecs, n_images):
    all_top_images = []
    for search_image_vec in search_image_vecs:
        top_images = get_similar_images(all_vecs, all_names, search_image_vec, n_images)
        all_top_images.append(top_images)

    all_top_images = pd.concat(all_top_images)

    # Remove duplicates
    all_top_images.drop_duplicates(subset=['name'], inplace=True)

    # Remove search images
    all_top_images = all_top_images[~all_top_images['name'].isin(search_image_names)]

    # Sort by distance
    all_top_images.sort_values(by='distance', inplace=True)
    all_top_images.reset_index(inplace=True)

    return all_top_images

all_top_images = get_all_similar_images(search_image_vecs, 30)

# %% Show matching images
def display_images(image_names, n_rows, n_cols, id, image_distances=None):
    # Make layout
    cols = []
    for _ in range(n_rows):
        rows = st.columns(n_cols)
        cols.extend(rows)

    checks = [None] * len(cols)
    with st.form(key=f'image-form-{id}'):
        for i, col in enumerate(cols):
            # Allow empty tiles
            if i < len(image_names):
                name = image_names[i]
                tile = col.container(height=350, border=True)
                tile.caption(f'{name}')

                if image_distances is not None:
                    distance = image_distances[i]
                    tile.caption(f'distance: {distance:.4f}')
                    checks[i] = tile.checkbox('selected', key=f'check-{id}-{i}')

                try:
                    tile.image(Image.open(path.join(st.session_state["filepath_images"], name + '.jpg')))
                except Exception as e:
                    st.warning(e)

        if image_distances is not None:
            submit = st.form_submit_button()
            if submit:
                selected_images = pd.Series(image_names[checks], name=f'selected images')
                st.caption('selected images:')
                st.dataframe(selected_images, hide_index=True)


# %% Show search images
n_cols = 4

image_names = search_image_names
n_rows = np.ceil(image_names.shape[0] / n_cols).astype(int)

st.subheader('Search images')
display_images(image_names, n_rows, n_cols,'search')

# %% Show similar images
n_cols = 4
n_rows = 10

top_image_names = all_top_images['name'].to_numpy()[:n_rows*n_cols]
top_image_distances = all_top_images['distance'].to_numpy()[:n_rows*n_cols]

st.subheader('Similar images')
display_images(top_image_names, n_rows, n_cols, 'top', top_image_distances)
