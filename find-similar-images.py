import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from glob import glob
from pyecotaxa.archive import read_tsv, write_tsv
from scipy.spatial.distance import cdist


# %%
# @st.cache_data
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
    top_images = distances.argsort()[range(n_images)]
    top_image_names = names[top_images]
    top_image_names = np.array([image_name.replace('.jpg', '') for image_name in top_image_names])
    top_image_distances = distances[top_images]
    return top_images, top_image_names, top_image_distances

top_images, top_image_names, top_image_distances = get_similar_images(all_vecs, all_names, search_image_names[50], 50)

# %%
# image_name = '2024-04-25_21-22-41-731183_8.jpg'
top_cols = st.columns(3)
st.session_state["disp_img"] = st.text_input("Image name", "2024-04-25_21-22-41-731183_8")

try:
    img = Image.open(path.join(st.session_state["filepath_in"], st.session_state["disp_img"] + '.jpg'))
    top_cols[1].image(img)
except Exception as e:
    st.warning(e)

n_rows = 25
n_cols = 4
cols = []
for _ in range(n_rows):
    rows = st.columns(n_cols)
    cols.extend(rows)

top_images, top_image_names, top_image_distances = get_similar_images(all_vecs, all_names, st.session_state["disp_img"], n_rows*n_cols)

# Show images
checks = [None] * len(cols)
with st.form(key='image-form'):
    for i, col in enumerate(cols):
        name = top_image_names[i]
        distance = top_image_distances[i]

        tile = col.container(height=350, border=True)
        tile.caption(f'{name}')
        tile.caption(f'distance: {distance:.4f}')
        try:
            # tile.image(Image.open(path.join(st.session_state["filepath_in"], names[top_images[i]])))
            tile.image(Image.open(path.join(st.session_state["filepath_in"], name + '.jpg')))
        except Exception as e:
            st.warning(e)
        checks[i] = tile.checkbox('selected', key=f'check-{i}')
    submit = st.form_submit_button()
    if submit:
        # st.write(checks)
        selected_images = pd.Series(top_image_names[checks], name=f'similar images for {name}')
        # selected_images = pd.Series(np.array(name, top_image_names[checks]), name='image names')
        # pd.concat([name, selected_images], ignore_index=True)
        st.caption('selected images:')
        st.dataframe(selected_images, hide_index=True)
