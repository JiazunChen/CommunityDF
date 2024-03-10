# CommunityDF
code for vldb24 'CommunityDF: A Guided Conditional Denoising Diffusion Approach for Community Search'


~~The code will be ready and released by the March 20th~~

## Quick Start

### Dependencies

This project is developed and tested on Ubuntu 20.04.6 LTS with Python 3.9.16. To see additional Python library dependencies, refer to the `requirements.txt` file.

We recommend using Anaconda to create a new environment:
```
conda create -n CommunityDF python=3.9.16
conda activate CommunityDF
pip install -r requirements.txt
```
 

### Datasets

We utilize six real-world benchmark datasets w, sourced from [SNAP](https://snap.stanford.edu/data/).

The processed  are stored in `dataset.7z`. Before first use, it is necessary to uncompress them:
```
7za x datasets.7z -oc:./datasets/
```

## Run

Default parameters are provided for quick and easy testing across all datasets (`dblp,amazon,youtube,facebook,twitter,lj`):

```
python main.py --facebook --train_size 30 --valid_size 10 --diffusion_steps -1
## The other datasets have a training set size of 450 and a validation set size of 50
## --diffusion_steps -1 means that the model will search for the best diffusion_step by iterating through [5,30] at intervals of 5
```

We can also specify a specific size:
```
python main.py --facebook --train_size 30 --valid_size 10 --diffusion_steps 10
```


By default, the best models are saved in the `./result/{time}` directory.