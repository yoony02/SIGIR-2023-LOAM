## LOAM
---- 

This is official Pytorch implementation of our paper **LOAM: Improving Long-tail Session-based Recommendation via Niche Walk  Augmentation and Tail Session Mixup**, accepted by *SIGIR '23*


## Datasets
---- 
Datasets we used in the paper can be downloaded from:

https://drive.google.com/drive/folders/1UnrR6w6dRQhnCIRCq0voN58cwwX5rHDV?usp=share_link

Unzip the datasets and move them to `Datasets/`. <br>
You can also preprocess raw datasets downloaded from public links by running `.ipynb` files in `Datasets/preprocess_code`.


## Requirements
----
- Python 3 
- PyTorch
- NetworkX
- Numpy 
- wandb



## Basic Usage
----
- Change the experimetal settings and model hyperparameters using the `config.py`
- Run `main.py --dataset [dataset_name]` to train and test models.
- You can record performance and loss values by setting `wandb`. 



## Citation
---
Please cite our paper if you use the code:
```

```
