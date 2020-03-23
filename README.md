# AmygNet

A 3D FCN with a top-down attention mechanism for segmenting extremely small brain structures, e.g., amygdala and its subnuclei.

![Att_image](https://github.com/YilinLiu97/AmygNet-Pytorch/blob/master/Fig_3.jpg)

# Citation
If you find the code here useful, please cite our paper:

Liu Y, Nacewicz BM, Zhao G, Adluru N, Kirk GR, Ferrazzano PA, Styner MA and Alexander AL (2020) A 3D Fully Convolutional Neural Network
With Top-Down Attention-Guided Refinement for Accurate and Robust Automatic Segmentation of Amygdala and Its Subnuclei. Front. Neurosci.14:260. doi: 10.3389/fnins.2020.00260
## Getting Started


### Prerequisites


```
Python 2.7
PyTorch 0.4.1
numpy 1.14.5
```

### Installation

(for Waisman users)
```
pip install torch==0.4.1 -f https://download.pytorch.org/whl/cu100/stable --user
pip install imgaug --user
pip install sklearn --user
```
### If you got this error: numpy.core.multiarray failed to import, do:
```
pip install numpy -l
```

### Path to the code
```
/study/utaut2/YL_AmygNet
```
### Data Preparation
Organize your folders as below:
```
Dataset/
   Training/  Labels/   Validation/
```

```
Validation/
   images/  labels/
```
```
Testing/
   images/  labels/ (if available)
```
   
### Training ('num_classes' should include background, i.e., N+1)
```
python train.py --sup_only True --data_path /path/to/Dataset --sourcefolder Training --labelfolder Labels --experiment_name XXX --num_classes XX --triple False --num_epochs XX
```
### Validation
```
python val.py --val_path /path/to/Dataset/Validation --valimagefolder images --vallabelfolder labels --model Test --num_gpus 3 --num_classes XX
```
### Testing, using the "best epoch" number shown after validation for "test_epoch"
```
python test.py --num_classes XX --save_path XXX --model XXX --test_path /path/to/Testing/images --test_epoch N 
```
### Evaluation
```
python metric.py /path/to/outputs_tobe_evaluated /path/to/groundtruths Number_Of_Classes
```

## Authors

* **Yilin Liu**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


