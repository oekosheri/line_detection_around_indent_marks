# Line detection around indentation marks
This package aims at detecting [slip lines](https://en.wikipedia.org/wiki/Slip_bands_in_metals) around indentation marks formed in mechanical testing of materials. To detect line segments using Opencv we found it easier to first detect/segment away the indentation mark from images. This helps up localize the regions of ineterst on the edges on the indentation mark where the slip lines usually appear. 

The [indent segmentation](https://github.com/oekosheri/line_detection_around_indent_marks/tree/main/indent_segmentation) runs a semnatic segmentation model on our images/masks with the goal of finding the best model to detect the indent mark. For details of this part refer to [README](https://github.com/oekosheri/line_detection_around_indent_marks/tree/main/indent_segmentation/README.md).   

