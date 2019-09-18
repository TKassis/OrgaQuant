# OrgaQuant
Intestinal Organoid Localization and Quantification Using Deep Convolutional Neural Networks

## Getting Started
### Prerequisites
You must have the following installed:
1. Python 3.6
2. Jupyter Notebook
3. Pandas
4. TensorFlow (with the appropriate CUDA version)

We recommend that you install Anaconda and run the code on an NVIDIA GPU with at least 4GB of GPU memory.

### Installation
1. Download or clone the OrgaQuant repository.
2. Download and unzip the trained intestinal organoid model from https://osf.io/dj4uk/

## Usage
1. Open the Jupyter Notebook called orgaquant_batch in the object_detection folder.
2. Modify the following to point to the correct files:

```python
PATH_TO_CKPT = "object_detection/organoid_inference_graph/frozen_inference_graph.pb"
PATH_TO_LABELS = "object_detection/data/organoid_label_map.pbtxt"
FOLDER_PATH = ""
look_pix = 500
slide_pix = 300
```
3. Run all the cells

When the script finishes running you should have a CSV file with the results as well as a labeled image file. If you encounter any problems please raise the issue here on Github.

## Contributing
If you have any improvements in mind please feel free to contribute to the project via GitHub. If you encounter any issue running the above code please raise the issue here.

## Citation
https://www.nature.com/articles/s41598-019-48874-y
