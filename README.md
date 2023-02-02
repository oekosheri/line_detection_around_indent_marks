# Line detection around indentation marks
This package aims at detecting [slip lines](https://en.wikipedia.org/wiki/Slip_bands_in_metals) around indentation marks formed in mechanical testing of materials. To detect line segments using Opencv we found it easier to first detect/segment away the indentation mark from images. This helps up localize the regions of ineterst on the edges on the indentation mark where the slip lines usually appear.

The [indent segmentation](https://github.com/oekosheri/line_detection_around_indent_marks/tree/main/indent_segmentation) runs a semnatic segmentation model on high performance computing cluster (HPC) with the goal of finding the best model to detect the indent mark. For details of this part I refer you to [README](https://github.com/oekosheri/line_detection_around_indent_marks/tree/main/indent_segmentation/README.md).

## Create the righ environment
The first step to use the line dtection code is to create the needed environment:
```
conda env create -f environment.yml
conda activate tensorflow-opencv2
```
## Run line detection
You could run the line detection file either locally or on the computer cluster.
To run it in a Terminal (ex. Linux), frist edit [run_detect_line.sh](./run_detect_line.sh) and change the address to the input image and the best model. Then run the file:
```
Zsh ./run_detect_line.sh
```
or directly run the python file []








