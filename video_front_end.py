import streamlit as st
import numpy as np
from PIL import Image
import time
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

    a1, a2, a3, a4 = st.columns(4)
    b1, b2, b3, b4 = st.columns(4)
    c1, c2, c3, c4 = st.columns(4)
    d1, d2, d3, d4 = st.columns(4)

    cols = (a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4)

    idx = int(np.argwhere(names == st.session_state["disp_img"]))
    target_vec = vecs[idx]


    distances = cdist(target_vec[None, ...], vecs, metric='cosine').squeeze()
    top_images = distances.argsort()[1:17]
    top_distances = distances[top_images]

    # Show images
    for i, col in enumerate(cols):
        name = names[top_images[0]].replace('.jpg', '')
        distance = top_distances[i]
        col.caption(f'{name}')
        col.caption(f'{distance:.4f}')
        col.image(Image.open(root + names[top_images[i]]))

except Exception as e:
    fcol3.caption('Invalid image name')
    pass


