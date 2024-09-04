# ctopt
A tool that is used to determine what is the optimal number of cell type clusters for given spatial transcriptomics sample using single cell reference. The tool leverages contrastive learning to map cell types from a single-cell reference dataset to a spatial transcriptomics dataset. 

## Installation
ctopt can be installed using pip:
`pip install ctopt`

## Usage
`ctopt` is intended to be used as a command-line tool.

usage: ```ctopt [-h] --sc_path SC_PATH --st_path ST_PATH -a ANNOTATION [-at ANNOTATION_ST]
             [--wandb_key WANDB_KEY] [--num_markers NUM_MARKERS] [--batch_size BATCH_SIZE]
             [--n_views N_VIEWS] [--epochs EPOCHS] [--emb_dim EMB_DIM] [--enc_depth ENC_DEPTH]
             [--class_depth CLASS_DEPTH] [--augmentation_perc AUGMENTATION_PERC]
             [--temperature TEMPERATURE] [-l] [-v]```

A script that performs reference based cell type annotation.
```
options:
  -h, --help            show this help message and exit
  --sc_path SC_PATH     A single cell reference dataset
  --st_path ST_PATH     A spatially resolved dataset
  -a ANNOTATION, --annotation ANNOTATION
                        Annotation label for cell types in single-cell reference dataset
  -at ANNOTATION_ST, --annotation_st ANNOTATION_ST
                        Annotation label for cell types in ST dataset if available
  --wandb_key WANDB_KEY
                        Wandb key for loss monitoring
  --num_markers NUM_MARKERS
                        Number of marker genes per cell type. Default is 100.
  --batch_size BATCH_SIZE
                        Number of samples in the batch. Default is 512.
  --n_views N_VIEWS     Number of views/augmentations of one sample(cell). Default and minimum is
                        2.
  --epochs EPOCHS       Contrastive: Number of epochs to train deep encoder. Default is 50.
  --emb_dim EMB_DIM     Contrastive: Dimension of the output embeddings. Default is 256.
  --enc_depth ENC_DEPTH
                        Number of hidden layers in the encoder MLP. Default is 1.
  --class_depth CLASS_DEPTH
                        Number of hidden layers in the classifier MLP. Specify 0 for logistic
                        regression. Default is 1.
  --augmentation_perc AUGMENTATION_PERC
                        Contrastive: Percentage for the augmentation of scRNA reference data. If
                        not provided it will be calculated automatically. Default is None.
  --temperature TEMPERATURE
                        Temperature value used in contrastive loss
  -l, --log_mem
  -v, --verbose         Enable logging by specifying --verbose
```



