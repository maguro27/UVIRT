# UVIRT
This repository contains the official PyTorch implementation of the following paper:
>[UVIRT—Unsupervised Virtual Try-on Using Disentangled Clothing and Person Features](https://www.mdpi.com/1424-8220/20/19/5647) <br>
>Hideki Tsunashima (Waseda Univ.), Kosuke Arase (Mercari, Inc.), Antony Lam (Mercari, Inc.), Hirokatsu Kataoka (AIST) <br>
>Sensors 2020

## Explanation of this prepository
- `gdwct`: Git submodule. We use [the official PyTorch implementation of the GDWCT paper](https://github.com/WonwoongCho/GDWCT).
- `gitmodules`: For git submodule.
- `data_loader_mpv.py`: Dataloader file for overwriting the gdwct dataloader.
- `googleDrive_download_folder.py`: Script using PyDrive for downloading the preprocessed MPV dataset.
- `mpv.yaml`: Config file for training UVIRT on the MPV dataset.
- `run_uvirt.py`: Main running script.

## Dependencies
This repository environment is as follow. We did not check the environment except for the following envirionment.
- `Python`: 3.6
- `PyTorch`: 1.6.0
- `Torchvision`: 0.7.0
- `cuda toolkit`: 10.2.89
- `cuDNN`: 8.0.2
- `GPU`: NVIDIA Tesla V100

Other requirements are in `requirements.txt` and can be installed with
```
pip install -r requirements.txt
```
***!!! Attention !!!*** <br>
The above script is destructive. For instance, you already install the numpy 1.19.5 and the requirements.txt describe `numpy==1.16.4`. The numpy is downgraded from 1.19.5 to 1.16.4. Therefore, please manually install the each modules if you want to avoid the destruction.

## Dataset
You can select the two types of downloading the preprocessed MPV dataset.
1. Manually download the dataset from GoogleDrive: [The preprocessed MPV dataset](https://drive.google.com/drive/folders/1oIpKLhc5Bwaz9IDobSGdk0UQRAuXZ-Wr?usp=sharing) <br>
You make the `datasets` directory and add the dataset into the `datasets` directory, after you complete downloading the dataset.
2. Use `googleDrive_download_folder.py`.
```
cd UVIRT
python googleDrive_download_folder.py -p 1oIpKLhc5Bwaz9IDobSGdk0UQRAuXZ-Wr -s ./datasets
cd datasets/mpv_preprocessed
mv ./* ../
rm -r mpv_preprocessed
```

## Test visualiation with pre-trained models on the MPV dataset
You can visualize virtual try-on results on the test set in the two steps.
1. Download the pre-trained models as well as downloading the dataset.
- Manually download the pre-trained models from GoogleDrive: [The pre-trained models](https://drive.google.com/drive/folders/1IZBOZX-vUxy3cI-SGmJ3mRgFHzI3DpKg?usp=sharing) <br>
You make the `./gdwct/models` directory and add the dataset into the `./gdwct/models` directory, after you complete downloading the pre-trained models. <br>
- Use `googleDrive_download_folder.py`.
```
cd UVIRT
python googleDrive_download_folder.py -p 1IZBOZX-vUxy3cI-SGmJ3mRgFHzI3DpKg -s ./gdwct/models/
```
2. Run the script.
```
run_uvirt.py -m test -l -s 490000 -model_save_path models/mpv/uvirt_pretrained_models
```

## Training
```
run_uvirt.py 
```

## Citation
```
@Article{s20195647,
AUTHOR = {Tsunashima, Hideki and Arase, Kosuke and Lam, Antony and Kataoka, Hirokatsu},
TITLE = {UVIRT—Unsupervised Virtual Try-on Using Disentangled Clothing and Person Features},
JOURNAL = {Sensors},
VOLUME = {20},
YEAR = {2020},
NUMBER = {19},
ARTICLE-NUMBER = {5647},
URL = {https://www.mdpi.com/1424-8220/20/19/5647},
ISSN = {1424-8220},
ABSTRACT = {Virtual Try-on is the ability to realistically superimpose clothing onto a target person. Due to its importance to the multi-billion dollar e-commerce industry, the problem has received significant attention in recent years. To date, most virtual try-on methods have been supervised approaches, namely using annotated data, such as clothes parsing semantic segmentation masks and paired images. These approaches incur a very high cost in annotation. Even existing weakly-supervised virtual try-on methods still use annotated data or pre-trained networks as auxiliary information and the costs of the annotation are still significantly high. Plus, the strategy using pre-trained networks is not appropriate in the practical scenarios due to latency. In this paper we propose Unsupervised VIRtual Try-on using disentangled representation (UVIRT). After UVIRT extracts a clothes and a person feature from a person image and a clothes image respectively, it exchanges a clothes and a person feature. Finally, UVIRT achieve virtual try-on. This is all achieved in an unsupervised manner so UVIRT has the advantage that it does not require any annotated data, pre-trained networks nor even category labels. In the experiments, we qualitatively and quantitatively compare between supervised methods and our UVIRT method on the MPV dataset (which has paired images) and on a Consumer-to-Consumer (C2C) marketplace dataset (which has unpaired images). As a result, UVIRT outperform the supervised method on the C2C marketplace dataset, and achieve comparable results on the MPV dataset, which has paired images in comparison with the conventional supervised method.},
DOI = {10.3390/s20195647}
}
```

## Acknowledgements
This repository use the official implementation of the paper "Image-to-Image Translation via Group-wise Deep Whitening-and-Coloring Transformation". This repository structure is inspired by [SPACE](https://github.com/zhixuan-lin/SPACE). We thank these repositor for help with the code release.
