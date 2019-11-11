# OrgaQuant
OrgaQuant is a simple-to-use python script with a user interface for measuring organoid number and diameter in 3D cultures.
While this repository contains an improved version of the code described in our [original paper](https://www.nature.com/articles/s41598-019-48874-y),
please refer to the manuscript for details. The original code described in the paper is available upon request
but is much slower than this current version. We highly recommend using the updated code available here.

![OrgaQuant for Measuring Organoid Diameter in 3D](/readme_images/Figure_1_From_Paper.jpg)

### Screenshot of the User Interface
![Screenshot of User Interface](/readme_images/screenshot.jpg)

## Getting Started
### Hardware Prerequisites
We recommend a computer with at least 16 GB of RAM and an NVIDIA graphics card (required) with at least 8 GB of GPU memory.
All code was developed and tested in a Windows environment, but should work fine on both Mac OS and Linux.

### Installation
1. Install Anaconda from https://www.anaconda.com/distribution/
2. Open the Anaconda Prompt and create a new conda environment using `conda create -n orgaquant python=3.6`
3. Activate the newly created environment using `activate orgaquant`
4. Install Tensorflow and Git using `conda install tensorflow-gpu=1.14 git`
5. Install dependencies using `pip install keras-resnet==0.2.0 cython keras matplotlib opencv-python progressbar2 streamlit`
6. Clone the OrgaQuant repository using `git clone https://github.com/TKassis/OrgaQuant.git`
7. Move into the directory using `cd OrgaQuant`
8. Install _keras_retinanet_ using `python setup.py build_ext --inplace`. More information [here](https://github.com/fizyr/keras-retinanet)
9. Download the pre-trained model from [here](https://github.com/TKassis/OrgaQuant/releases/download/v0.1/orgaquant_intestinal_v2.h5) and place inside the _trained_models_ folder.

## Usage
1. Within the OrgaQuant directory run the following: `streamlit run orgaquant.py`. This should automatically open a browser window with the user interface.
2. Indicate the folder that contains the organoid images (default is _/test_folder_)
3. Modify the sliders to adjust some of the settings based on your images.
4. You can batch process all your images by clicking on 'Process All'. This will use the same settings you adjusted for the sample image.

When the script finishes running you should have a CSV file with the results as well as a labeled image file for each image in the folder.
If you encounter any problems please raise the issue here on [Github](https://github.com/TKassis/OrgaQuant/issues).

## Custom Models
If you have certain types of organoids for which the trained model provided here does not work we might be able to create a new trained model for you and make it available here if you share some of your images with us.
Please raise the request as a [Github Issue](https://github.com/TKassis/OrgaQuant/issues). If you would like to train your own model, you may follow the instructions given for the
[Keras RetinaNet package](https://github.com/fizyr/keras-retinanet).

## Contributing
If you have any improvements in mind please feel free to contribute to the project via GitHub. If you encounter any issue running the above code please raise the issue [here](https://github.com/TKassis/OrgaQuant/issues).

## Citation
If you use this code, please cite:
```
Kassis et al., OrgaQuant: Human Intestinal Organoid Localization and Quantification Using Deep Convolutional Neural Networks. Sci. Rep. 9, 1–7 (2019).
```
The open access paper can be accessed through https://www.nature.com/articles/s41598-019-48874-y.

OrgaQuant builds on the object detection framework in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Lin et al., 2017.
