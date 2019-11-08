# OrgaQuant
OrgaQuant is a simple to use script for measuring organoid number and diameter in 3D cultures. While this repository contains an improved version of the code described in our original paper, please refer
to the manuscript for details. The original code described in the paper is available upon request but is much slower than this current version. We recommend you use the update code avaialble here.

![Screenshot of User Interface](/readme_images/screenshot.jpg)

## Getting Started
### Hardware Prerequisites
We recommend an NVIDIA GPU with at least 8 GB of GPU memory. Other configurations might work but are not tested.

### Installation
1. Install Anaconda from https://www.anaconda.com/distribution/
2. Create a new conda environment using `conda create -n orgaquant python=3.6`
3. Activate the newly created environment using `activate orgaquant`
4. Install Tensorflow using `conda install tensorflow-gpu=1.14`
5. Install git using `conda install git`
6. Install dependencies using `pip install keras-resnet==0.2.0 cython keras matplotlib opencv-python progressbar2`
7. Install Streamlit using `pip install streamlit`
8. Clone this repository using `git clone https://github.com/TKassis/OrgaQuant.git`
9. Move into the directory using `cd OrgaQuant`
10. Install _keras_retinanet_ using `python setup.py build_ext --inplace`. More information here: https://github.com/fizyr/keras-retinanet
11. Create a new folder called _trained_models_ and place in it the trained model from https://github.com/TKassis/OrgaQuant/releases/download/v0.1/orgaquant_intestinal_v2.h5

## Usage
1. Within the OrgaQuant directory run the following: `streamlit run orgaquant.py`
2. Indicate the folder that contains the images
3. Modify the sliders to adjust some of the settings based your images.
4. You can batch process all your images by clicking on 'Process All'. This will use the same settings you adjusted for the sample image.

When the script finishes running you should have a CSV file with the results as well as a labeled image file. If you encounter any problems please raise the issue here on Github.

## Contributing
If you have any improvements in mind please feel free to contribute to the project via GitHub. If you encounter any issue running the above code please raise the issue here.

## Citation
https://www.nature.com/articles/s41598-019-48874-y
