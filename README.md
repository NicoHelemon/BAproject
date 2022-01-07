# BAproject

Code repository for my bachelor semester project *Using Deep Features for Segmentation Annotations*

## Dependencies

non-exhaustively:

* Python 3.6
* OpenCV
* Pil
* NumPy
* Pytorch

## Subdirectories and files desctiptions

* `data_cleaning` - Contains data cleaning and json generating code.
* `grabcut_MoetaYuko` - Auxiliary and aborted. A copy of [MoetaYuko's GrabCut implementation](https://github.com/MoetaYuko/GrabCut/blob/master/README.md). There was a small attempt to build a *[Grabcut: Interactive foreground extraction using iterated graph cuts, ACM SIGGRAPH 2004](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)* and *[CAM](https://arxiv.org/abs/1512.04150)* based segmentation technique (using MoetaYuko's implementation). Ultimately, there is no concrete result.
* `json` - Contains json and csv files.
* `output` - Contains the result of the different segmentation techniques on PASCAL VOC 2012 ; mainly csv files and graphs.
* `utils` - Contains the main code: the different segmentation techniques and utility functions.
* `visualization` - Contains a jupyter notebook to see the segmentation techniques in action.
* `Class_segmentation.py`, `Object_segmentation.py` - Files running the class and object segmentation techniques on PASCAL VOC 2012 and saving the results in the `output` directory. Caution! The results take a very long time to be generated (about 2-6 hours).
