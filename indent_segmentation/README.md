## Segmentation of indent marks
In this section a semantic segmentation model based on the unet algorithm will be used to segment the indentation marks. The data, including images and masks, that were used for training is available as open data [here](). Images are gathered from different scanning electron microscopes.

Training and submission scripts have been prepared for use on an HPC cluster (In this case [RWTH University ITC]()). To find the best model, we submitted multiple jobs with different input arguments, in other words we ran a grid search.

To run the submission files as is, simply run:
```
zsh submission.sh
```
For each job a directory will be created which contains submit files, output files, log files and models. You can edit the grid to be searched in [submission file](submission.sh), [run_file](run_file.sh]), [sub_file](sub_file.sh]) and finally by input arguments in [training file](training.py).



