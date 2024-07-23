import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist


root = "/Users/kevinadmin/Desktop/PlanktoScope Processing/Test/export_12581_20240719_1809/LUMCON Oyster Larvae Sampling 2024-04-25_1/"

@st.cache_data
def read_data():
    all_vecs = np.load(f"{root}/all_vecs.npy")
    all_names = np.load(f"{root}/all_names.npy")
    return all_vecs, all_names


vecs, names = read_data()

_, fcol2, fcol3, _, _ = st.columns(5)

# image_name = '2024-04-25_21-22-41-731183_8.jpg'
image_name = st.text_input("Image name", "2024-04-25_21-22-41-731183_8")
image_name = image_name + '.jpg'
try:
    img = Image.open(f"{root}/{image_name}")
    st.session_state["disp_img"] = image_name

    fcol3.image(Image.open(root + st.session_state["disp_img"]))
    fcol3.caption(st.session_state["disp_img"])

    n_rows = 25
    n_cols = 4
    cols = []
    for _ in range(n_rows):
        rows = st.columns(n_cols)
        cols.extend(rows)

    idx = int(np.argwhere(names == st.session_state["disp_img"]).squeeze())
    target_vec = vecs[idx]

    distances = cdist(target_vec[None, ...], vecs, metric='cosine').squeeze()
    top_images = distances.argsort()[range(n_rows*n_cols)]
    top_image_names = names[top_images]
    top_image_names = np.array([image_name.replace('.jpg', '') for image_name in top_image_names])
    top_distances = distances[top_images]

    # Show images
    checks = [None] * len(cols)
    with st.form(key='image-form'):
        for i, col in enumerate(cols):
            name = top_image_names[i]
            distance = top_distances[i]

            tile = col.container(height=350, border=True)
            tile.caption(f'{name}')
            tile.caption(f'distance: {distance:.4f}')
            tile.image(Image.open(root + names[top_images[i]]))
            checks[i] = tile.checkbox('selected', key=f'check-{i}')
        submit = st.form_submit_button()
        if submit:
            # st.write(checks)
            selected_images = pd.Series(top_image_names[checks], name=f'similar images for {name}')
            # selected_images = pd.Series(np.array(name, top_image_names[checks]), name='image names')
            # pd.concat([name, selected_images], ignore_index=True)
            st.caption('selected images:')
            st.dataframe(selected_images, hide_index=True)

except Exception as e:
    # fcol3.caption('Invalid image name')
    fcol3.caption(e)
    pass


