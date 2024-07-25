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
def read_data(root):
    all_vecs = np.load(f"{root}/all_vecs.npy")
    all_names = np.load(f"{root}/all_names.npy")
    all_names = np.array([image_name.replace('.jpg', '') for image_name in all_names])
    search_image_path = glob(f"{root}/data/ecotaxa*.tsv")[0]
    search_image_data = read_tsv(search_image_path)
    search_image_names = search_image_data['object_id'].to_numpy()
    search_image_names = np.array([image_name.replace('.jpg', '') for image_name in search_image_names])

    return all_vecs, all_names, search_image_names

st.title('Plankton picker')

# st.session_state["filepath_in"] = st.text_input("File path", "")
st.session_state["filepath_in"] = st.text_input("File path", "/Users/kevinadmin/Desktop/Image Similarity/LUMCON Oyster Larvae Sampling 2024-04-25_1")

# Get data files
try:
    all_vecs, all_names, search_image_names = read_data(st.session_state["filepath_in"])
    st.subheader(st.session_state["filepath_in"])
except Exception as e:
    st.warning(e)

# %%
def get_similar_images(vecs, names, search_image_name, n_images):
    search_image_name = search_image_name
    idx = int(np.argwhere(all_names == search_image_name).squeeze())
    target_vec = vecs[idx]
    distances = cdist(target_vec[None, ...], vecs, metric='cosine').squeeze()
    top_image_indices = distances.argsort()[range(n_images)]
    top_image_names = names[top_image_indices]
    top_image_names = np.array([image_name.replace('.jpg', '') for image_name in top_image_names])
    top_image_distances = distances[top_image_indices]
    top_images = pd.DataFrame(data={'index': top_image_indices, 'name': top_image_names, 'distance': top_image_distances})
    return top_images


# %% Get similar images for all search images
@st.cache_data
def get_all_similar_images(search_image_names, n_images):
    all_top_images = []
    for search_image_name in search_image_names:
        top_images = get_similar_images(all_vecs, all_names, search_image_name, n_images)
        all_top_images.append(top_images)

    all_top_images = pd.concat(all_top_images)

    # Remove duplicates
    all_top_images.drop_duplicates(subset=['name'], inplace=True)

    # Sort by distance
    all_top_images.sort_values(by='distance', inplace=True)
    all_top_images.reset_index(inplace=True)
    return all_top_images

all_top_images = get_all_similar_images(search_image_names, 30)

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
            name = image_names[i]
            tile = col.container(height=350, border=True)
            tile.caption(f'{name}')

            if image_distances is not None:
                distance = image_distances[i]
                tile.caption(f'distance: {distance:.4f}')

            try:
                tile.image(Image.open(path.join(st.session_state["filepath_in"], name + '.jpg')))
            except Exception as e:
                st.warning(e)
            checks[i] = tile.checkbox('selected', key=f'check-{id}-{i}')
        submit = st.form_submit_button()
        if submit:
            selected_images = pd.Series(image_names[checks], name=f'selected images')
            st.caption('selected images:')
            st.dataframe(selected_images, hide_index=True)


# %% Show search images
n_rows = 10
n_cols = 4

# top_image_names = all_top_images['name'].to_numpy()[:n_rows*n_cols]
# top_image_distances = all_top_images['distance'].to_numpy()[:n_rows*n_cols]
st.subheader('Search images')
display_images(search_image_names, n_rows, n_cols,'search')

# %% Show similar images
n_rows = 10
n_cols = 4

top_image_names = all_top_images['name'].to_numpy()[:n_rows*n_cols]
top_image_distances = all_top_images['distance'].to_numpy()[:n_rows*n_cols]

st.subheader('Similar images')
display_images(top_image_names, n_rows, n_cols, 'top', top_image_distances)
