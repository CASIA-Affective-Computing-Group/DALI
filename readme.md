# DALI: Dynamically Adjusted Label Importance for Noisy Partial Label Learning 

Correspondence to: 
  - Mingyu Xu*  (xumingyu2021@ia.ac.cn)
  - Zheng Lian* (lianzheng2016@ia.ac.cn)

## Paper
[**DALI: Dynamically Adjusted Label Importance for Noisy Partial Label Learning**](https://arxiv.org/pdf/2301.12077.pdf)<br>
Mingyu Xu *, Zheng Lian *, Lei Feng, Bin Liu, Jianhua Tao<br>

Please cite our paper if you find our work useful for your research:

```tex
@article{xu2023dali,
  title={DALI: Dynamically Adjusted Label Importance for Noisy Partial Label Learning},
  author={Xu, Mingyu and Lian, Zheng and Feng, Lei and Liu, Bin and Tao, Jianhua},
  journal={arXiv preprint arXiv:2301.12077},
  year={2023}
}
```

## Usage

### Download Datasets

1. CIFAR10 and CIFAR100 can be downloaded automatically during program operation.
2. For CUB200, PiCO provides a preprocessed copy at [link](https://drive.google.com/file/d/1KNMPuKT1q3a6zIEgStar2o4xjs_a3Kge/view?usp=sharing) and just put the files to dataset/cub200/processed. See [link](https://github.com/hbzju/PiCO) for detail.



### Run DALI

We provide shell codes for model training. We do not tune the optimal parameters. Adjust the learning rate, epochs, and the time to start applying DALI will achieve better performance. 

~~~~shell
1.Run CIFAR10 with q=0.3 \eta=0.3:
python -u train_merge.py --seed=1 --save_root=savefinals --dataset=cifar10 --partial_rate=0.3 --noise_rate=0.3 --epochs=1000 --encoder=resnet --lr=0.01 --lr_adjust=Case1 --optimizer=sgd --weight_decay=1e-3 --gpu=1  --piror_start 80   --max1

2.Run CIFAR100 with q=0.05 \eta=0.3:
python -u train_merge.py --seed=1 --save_root=savefinals --dataset=cifar100 --partial_rate=0.05 --noise_rate=0.3 --epochs=1000 --encoder=resnet --lr=0.01 --lr_adjust=Case1 --optimizer=sgd --weight_decay=1e-3 --gpu=1  --piror_start 80   --max1

3.Run CUB200 with q=0.05 \eta=0.2:
python -u train_merge.py --seed=1 --save_root=savefinals --dataset=cub200 --partial_rate=0.05 --noise_rate=0.2 --epochs=200 --encoder=resnet --lr=0.01 --lr_adjust=Case1 --optimizer=sgd --weight_decay=1e-5 --gpu=4 --proto_start 30 --batch_size=256 --moco_queue=4096 --piror_start 60   --max1

4.Run CIFAR100H with q=0.5 \eta=0.2:
python -u train_merge.py --seed=1 --save_root=savefinals --dataset=cifar100H --partial_rate=0.5 --noise_rate=0.2 --epochs=1000 --encoder=resnet --lr=0.01 --lr_adjust=Case1 --optimizer=sgd --weight_decay=1e-3 --gpu=5 --piror_start 80   --max1

5.Run CIFAR100 with q=0.05 \eta=0.3 without mixup:
python -u train_merge.py --seed=1 --save_root=savefinals --dataset=cifar100 --partial_rate=0.05 --noise_rate=0.3 --epochs=1000 --encoder=resnet --lr=0.01 --lr_adjust=Case1 --optimizer=sgd --weight_decay=1e-3 --gpu=1  --piror_start 80   --loss_weight_mixup 0 --max1

6.Run CIFAR100 with q=0.05 \eta=0.3 with fix \lambda=0.4:
python -u train_merge.py --seed=1 --save_root=savefinals --dataset=cifar100 --partial_rate=0.05 --noise_rate=0.3 --epochs=1000 --encoder=resnet --lr=0.01 --lr_adjust=Case1 --optimizer=sgd --weight_decay=1e-3 --gpu=0  --piror_start 80   --piror_auto='case3' --piror_add=0.4 --piror_max=0.4 --max1
~~~~



