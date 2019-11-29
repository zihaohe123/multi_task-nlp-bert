# multi_task-nlp-bert
This is a repo for NLP multi-task learning, which includes single-sentence classification, pairwise text similarity, pairwise text classification, and relevance ranking.

## Downloading Datesets
Create a directory data/.
```
mkdir data
```

Download [SST-2.zip](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8) to data/SST-2.zip.

Download [STS-B.zip](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5) to data/STS-B.zip.

Download [QNLIv2.zip](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601) to data/QNLIv2.zip.

All the three datasets will be unzipped automatically so don't unzip these three files.

SNLI dataset will be downloaded automatically.


## Experiments
This is a multi-task learning so data would take a lot of memory, so we should use a small batch size.

Train with max_grad_norm 1.0 with multi-task learning.
```
python -u main.py --batch_size=16 --gpu=0,1 --grad_max_norm=1 --multi_task
python -u main.py --batch_size=8 --gpu=0 --grad_max_norm=1 --multi_task
```

Train without multi-task learning. Now we have less datasets to train so we could increase the batch size.
```
python -u main.py --batch_size=80 --gpu=0,1
python -u main.py --batch_size=40 --gpu=0,1

```