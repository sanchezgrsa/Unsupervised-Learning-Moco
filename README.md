# Unsupervised_Learning_Moco

This is an Unofficial implementation of [MoCo](https://arxiv.org/abs/1911.05722) on Pytorch using CIFAR-10.  

The encoder is calculated following the MoCo implementation. On each iteration it is fed to a linear neural network for classification purposes. 

## HYPERPARAMETERS

lr: 'initial learning rate',
epochs: 'number of total epochs to run'
schedule:'learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on'
cos:'use cosine lr schedule'

batch-size: 'mini-batch size'
wd: 'weight decay'

###### moco specific configs:
moco-dim: 'feature dimension'
moco-k: 'queue size; number of negative keys
moco-m: 'moco momentum of updating key encoder'
moco-t: 'softmax temperature'

bn-splits: 'simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu'

symmetric: 'use a symmetric loss function that backprops to both crops'

###### knn monitor
knn-k: 'k in kNN monitor'
knn-t: 'softmax temperature in kNN monitor; could be different with moco-t'

###### utils
resume: 'path to latest checkpoint (default: none)'
results-dir: 'path to cache (default: none)'
