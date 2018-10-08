# OrgaQuant
Intestinal Organoid Localization and Quantification Using Deep Convolutional Neural Networks

## Getting Started
### Prerequisites
You must have TensorFlow installed.

### Installation
Download or clone the OrgaQuant repository.
Download and unzip the trained intestinal organoid model from https://osf.io/dj4uk/

## Usage
Open the Jupyter Notebook called orgaquant_batch in the object_detection folder.
Modify the following to point to the correct files:

'''python

PATH_TO_CKPT = "C:/Users/Timothy/Desktop/OD/object_detection/organoid_inference_graph/frozen_inference_graph.pb"
PATH_TO_LABELS = "C:/Users/Timothy/Desktop/OD/object_detection/data/organoid_label_map.pbtxt"
FOLDER_PATH = "C:/Users/Timothy/Desktop/OrgaQuant Paper/Example Analysis/Ana/"
look_pix = 600
slide_pix = 600

'''
## Contributing
If you have any improvements in mind please feel free to contribute to the project via GitHub. If you encounter any issue running the above code please raise the issue here.

## Citation
Please cite our paper when this code is used along with the Object Detection API
