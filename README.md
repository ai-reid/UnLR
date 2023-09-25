# UnLR
Offical implementation of our "Uncertainty-guided Robust Labels Refinement for Unsupervised Person Re-identification" in NCAA2023 by Chengjun Wang, Jinjia Peng, Zeze Tao and Huibing Wang.

## Requirements
- Python 3.6+
- Pytorch 1.9.0
- For more detailed requirements, run
  ```
  pip install -r requirements.txt
  ```

## Dataset prepration
The file structure of the datasets for training is shown as below.<br>
```
/path/to/your/dataset
├── market1501
│   │── Market-1501-v15.09.15
│   │   │── bounding_box_test
│   │   └── bounding_box_train
│   │   └── query
│   │   └── ...
├── msmst17
│   │── MSMT17
│   │   │── bounding_box_test
│   │   └── bounding_box_train
│   │   └── query
│   │   └── ...
├── personx<br>
│   │── PersonX<br>
│   │   │── bounding_box_test
│   │   └── bounding_box_train
│   │   └── query
│   │   └── ...
```
## Train 
```
python examples/uncertain_main.py
```
## Test
```
python examples/test.py
```
