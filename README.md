## Feature based loss function for retinal multi-class vessel segmentation and feature measurement


### To do list

Add more description and binary vessel segmentation.


### Pretrained Model

The pretrained models in [Google_DRIVE](https://drive.google.com/file/d/1HDquRTSafMJE-9KfhDRBfGpZgPbXrRCC/view?usp=sharing). Download them and unzip them.


### Train

Start training, the dataset can be set as DRIVE_AV, LES-AV, or HRF-AV.
```
python train.py --e=1000 \
                --batch-size=2 \
                --learning-rate=8e-4 \
                --v=10.0 \
                --alpha=0.5 \
                --beta=1.0 \
                --gama=1.0 \
                --dataset='DRIVE_AV' \
                --loss='CF' \
                --uniform=True
```

### Test
Test the trained models.
```
python test.py --batch-size=1 \
                --uniform=True \
                --test_dataset='DRIVE_AV' \
                --loss='CF'

```


## Citation

TBC

