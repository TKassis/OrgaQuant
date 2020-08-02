# import dependencies
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image, adjust_contrast
from keras_retinanet.utils.visualization import draw_box
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title('OrgaQuant: Organoid Quantification')

st.sidebar.header('Settings')

model_path = st.sidebar.text_input("Saved model to use (stored in train_models folder):", value='orgaquant_intestinal_v3')

st.cache(allow_output_mutation=True)
def load_orga_model():
    return models.load_model(os.path.join('trained_models', model_path + '.h5'), backbone_name='resnet50')

model = load_orga_model()


folder_path = st.sidebar.text_input("Path to folder containing images:", value='test_folder')

imagelist=[]

for root, directories, filenames in os.walk(folder_path):
    imagelist = imagelist + [os.path.join(root,x) for x in filenames if x.endswith(('.jpg','.tif','.TIF', '.png', '.jpeg', '.tiff'))]

sample_image = st.sidebar.slider("Sample Image:", min_value=0, max_value=len(imagelist), step=1, value=0)
min_side = st.sidebar.slider("Image Size:", min_value=800, max_value=2000, step=100, value=1200)
st.sidebar.text('Larger "Image Size" allows you to detect smaller orgaoids at the cost of computational demand.')
contrast = st.sidebar.slider("Contrast:", min_value=1.0, max_value=3.0, step=0.25, value=1.5)
st.sidebar.text('Larger "Contrast" can improve detection sometimes.')
threshold = st.sidebar.slider("Confidence Threshold:", min_value=0.0, max_value=1.0, step=0.05, value=0.85)
st.sidebar.text('Use larger "Threshold" to eliminate false positives.')

# load image
#image = read_image_bgr(os.path.join(folder_path, imagelist[sample_image]))
image = read_image_bgr(imagelist[sample_image])

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = adjust_contrast(image,contrast)
image = preprocess_image(image)
image, scale = resize_image(image, min_side=min_side, max_side=2048)

# process image
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

# correct for image scale
boxes /= scale

num_org = 0
# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < threshold:
        break
    num_org= num_org + 1
    b = box.astype(int)
    draw_box(draw, b, color=(255, 0, 255))

st.image(draw,use_column_width=True)
st.write("Image name:", imagelist[sample_image])
st.write("Number of organoids detected:", num_org)

# Batch process from here on
st.sidebar.subheader('Batch Processing')

if st.sidebar.button("Process All"):
    progress_bar = st.sidebar.progress(0)
    for i, filename in enumerate(imagelist):
        try:
            #IMAGE_PATH = os.path.join(root,filename)
            IMAGE_PATH = filename
            # load image
            image = read_image_bgr(IMAGE_PATH)

            # copy to draw on
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # preprocess image for network
            image = adjust_contrast(image,contrast)
            image = preprocess_image(image)
            image, scale = resize_image(image, min_side=min_side, max_side=2048)

            # process image
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

            # correct for image scale
            boxes /= scale

            out = np.empty((0,4), dtype=np.float32)

            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < threshold:
                    break
                out = np.append(out, box.reshape(1,4), axis=0)

                b = box.astype(int)
                draw_box(draw, b, color=(255, 0, 255))

            output = pd.DataFrame(out,columns=['x1', 'y1', 'x2', 'y2'], dtype=np.int16)
            output['Diameter 1 (Pixels)'] = output['x2'] - output['x1']
            output['Diameter 2 (Pixels)'] = output['y2'] - output['y1']
            output.to_csv(IMAGE_PATH + '.csv', index=False)
            plt.imsave(IMAGE_PATH + '_detected.png', draw)
            progress_bar.progress((i+1)/len(imagelist))
        except:
            pass

    st.success('Analysis complete!')

else:
    st.sidebar.text("Click above to process all images.")

st.sidebar.markdown('If you find this helpful for your research please cite: \
_[OrgaQuant: Human Intestinal Organoid Localization and Quantification Using Deep Convolutional Neural Networks](https://www.nature.com/articles/s41598-019-48874-y)_.')

st.sidebar.markdown('For assistance or to report bugs, please raise an issue on [GitHub](https://github.com/TKassis/OrgaQuant/issues)')
