# Unsupervised_Learning_Moco

This is an Unofficial implementation of [MoCo](https://arxiv.org/abs/1911.05722) on Pytorch using CIFAR-10.  

The encoder is calculated following the MoCo implementation. On each iteration it is fed to a linear neural network for classification purposes. 

## HYPERPARAMETERS

lr: 'initial learning rate' <br/>
epochs: 'number of total epochs to run' <br/>
schedule:'learning rate schedule (when to drop lr by 10x), does not take effect if --cos is on' <br/>
cos:'use cosine lr schedule' <br/>

batch-size: 'mini-batch size' <br/>
wd: 'weight decay' <br/>

###### configurations specific to the MoCo model:
moco-dim: 'feature dimension' <br/>
moco-k: 'queue size; number of negative keys <br/>
moco-m: 'moco momentum of updating key encoder' <br/>
moco-t: 'softmax temperature' <br/>

bn-splits: 'simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu' <br/>

symmetric: 'use a symmetric loss function that backprops to both crops' <br/>


###### utils
resume: 'path to latest checkpoint (default: none)' <br/>
results-dir: 'path to cache (default: none)' <br/>
