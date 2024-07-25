import numpy as np
import pandas as pd
from glob import glob
from os import path

# import re
# from datetime import datetime

# match = re.search(r'\d{4}-\d{2}-\d{2}', text)
# date = datetime.strptime(match.group(), '%Y-%m-%d').date()

# %% Get data
filepath_in = '/Users/kevinadmin/Desktop/Image Similarity/LUMCON Oyster Larvae Sampling 2024-04-25_1/data'
files = glob(path.join(filepath_in, '*.csv'))

# %% Get data
data_list = []
image_list = []
for file in files:
    data = pd.read_csv(file)
    # data_list.append(data)
    images = data.iloc[:, 0].to_list()
    image_list.extend(images)

# %% Remove duplicates
image_list = np.array(image_list)
image_list = np.unique(image_list)

image_list = pd.Series(image_list, name='').sort_values()

filepath = path.join(filepath_in, 'similar_images.csv')
image_list.to_csv(filepath, index=False)
