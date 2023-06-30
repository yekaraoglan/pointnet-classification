# pointnet-classification

## Overview
This is a [PyTorch](https://github.com/pytorch/pytorch) implementation of [PointNet](https://arxiv.org/abs/1612.00593) for classification on ModelNet10 dataset.

## Requirements
Requirements are listed in `requirements.txt`.
```
pip install -r requirements.txt
```

## Usage
### Data Preparation
Download the dataset from [here](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset?resource=download) and unzip it to `data/` folder.

Example of a chair from the dataset:
![Example chair](figure.png)

### Folder Structure
```
├── data
|   ├── metadata_modelnet10.csv
│   ├── ModelNet10
│   │   │   ├── bathtub
|   |   |   |   ├── bathtub_0001.off
|   |   |   |   ├── ...
│   │   │   ├── bed
│   │   │   ├── ...
├── main.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
```


### Training
```
python main.py
```

### Testing
```
python main.py --mode test
```


## References
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [Another PyTorch implementation of PointNet](https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch/notebook) (util functions are borrowed from this notebook)
- [ModelNet10 dataset](https://modelnet.cs.princeton.edu/)
- [ModelNet10 dataset on Kaggle](https://www.kaggle.com/balraj98/modelnet10-princeton-3d-object-dataset)

### Citation
```
@misc{qi2016pointnet,
      title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation}, 
      author={Charles R. Qi and Hao Su and Kaichun Mo and Leonidas J. Guibas},
      year={2016},
      eprint={1612.00593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```