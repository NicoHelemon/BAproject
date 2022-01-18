# BAproject

Code repository for my bachelor semester project *Using Deep Features for Segmentation Annotations*.

# Description

3 segmentations techniques are tested on the PASCAL VOC 2012 dataset.

* *[GrabCut](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)* (OpenCV's implementation)
* *[CAM](https://arxiv.org/abs/1512.04150)* reliant
* GrabCut and CAM reliant 

## Dependencies

non-exhaustively:

* Python 3.6
* OpenCV
* Pil
* NumPy
* Pytorch

## Subdirectories and files desctiptions

* `data_cleaning` - Contains data cleaning and json generating code. **Contains deprecated code (_w.r.t._ the rest of the project)
* `grabcut_MoetaYuko` - Auxiliary and aborted. A copy of [MoetaYuko's GrabCut implementation](https://github.com/MoetaYuko/GrabCut/blob/master/README.md). There was a small attempt to merge GrabCut and CAM in a more clever and connected way. It would have been based on MoetaYuko's implementation. Ultimately, there is no concrete result. **Contains deprecated code (_w.r.t._ the rest of the project)
* `json` - Contains json and csv files.
* `output` - Contains the result of the different segmentation techniques on PASCAL VOC 2012 ; mainly csv files and graphs.
* `utils` - Contains the main code: the different segmentation techniques and utility functions.
* `visualization` - Contains a jupyter notebook to see the different segmentation techniques in action.
* `Class_segmentation.py`, `Object_segmentation.py` - Files running the class and object segmentation techniques on PASCAL VOC 2012 and saving the results in the `output` directory. Caution! The results take a very long time to be generated (about 2-6 hours).

## Visualization

Check `Visualization.ipynb` file in `visualization` directory.

## Utils

Only important functions have "proper" docummentation (description, paramaters, returned objects)

Main files:

* `cam.py` - Contains Class Activation Map related code and CamExtension and Cam classes definitions.
* `grabcut.py` - Contains GrabCut related code.
* `metrics.py` - Contains metrics definitions.
* `segmentation.py` - Contains the segmentation techniques.

Utility files:

* `image.py` - Contains image support functions.
* `json.py` - Contains json support functions.
* `path.py` - Contains path support functions.
* `VOCSegmentation.py` - Contains a personal extension of (https://pytorch.org/vision/main/generated/torchvision.datasets.VOCSegmentation.html).


