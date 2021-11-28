# MEMORABLE


This repository contains the source code accompanying our ICCV 2021 paper.

**[A Machine Teaching Framework for Scalable Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_A_Machine_Teaching_Framework_for_Scalable_Recognition_ICCV_2021_paper.html)**  
[Pei Wang](http://www.svcl.ucsd.edu/~peiwang), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno).  
In ICCV, 2021.

```
@InProceedings{wang2021gradient,
author = {Wang, Pei and Vasconcelos, Nuno},
title = {A Machine Teaching Framework for Scalable Recognition},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
}
```

## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 1.0. Other versions should work after minor modification.
2. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.


## Datasets

[Butterflies and Chinese Characters](https://github.com/macaodha/explain_teach/tree/master/data), [Gull](https://drive.google.com/file/d/1gjt2GIiGtvpsUoQywktP6RE0KhHunAzI/view?usp=sharing) are used. Please organize them as below after download,

```
datasets
|_ butterflies_crop
  |_ images
    |_ Viceroy
    |_ ...
|_ chinese_chars
  |_ images
    |_ grass
    |_ ...
|_ CUBgull
  |_ images
    |_ CaliforniaGull
    |_ ...
```



## Implementation details

### To generate counterfactual explanations
```
get_all_CE_butterflies.py
get_all_CE_gull.py
```
by SimCLR model
```
get_all_CE_butterflies_simclr.py
get_all_CE_gull_simclr.py
```
### To generate teaching images and explanations
```
train_butterflies_CMaxGrad.py
train_gull_CMaxGrad.py
```
by SimCLR model
```
train_butterflies_CMaxGrad_simclr.py
train_gull_CMaxGrad_simclr.py
```

For questions, feel free to reach out
```
Pei Wang: peiwang062@gmail.com
```

