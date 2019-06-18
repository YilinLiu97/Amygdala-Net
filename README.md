# AmygNet

A segmentation model for extremely small brain structures, e.g., amygdala and its subnuclei.

## Getting Started


### Prerequisites


```
Python 2.7
PyTorch 0.4.1
numpy 1.14.5
```

### Installing

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
### Preparation for the data
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
   images/  labels/ (if avilable)
```
   
### Training
```
python train.py --sup_only True --data_path /path/to/Dataset--sourcefolder Training --labelfolder Labels --experiment_name XXX --num_classes XX --triple False --num_epochs XX
```
### Validation
```
python val.py --val_path /path/to/Dataset/Validation --valimagefolder images --vallabelfolder labels --model Test --num_gpus 3 --num_classes 11
```
### Testing
```
python test.py --num_classes XX --save_path XXX --model XXX --test_path /path/to/Testing/images --test_epoch N (use the best epoch shown during validation)
```

## Authors

* **Yilin Liu**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


