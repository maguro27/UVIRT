# UVIRT
This repository contains the official PyTorch implementation of the following paper:
>[UVIRT—Unsupervised Virtual Try-on Using Disentangled Clothing and Person Features](https://www.mdpi.com/1424-8220/20/19/5647) <br>
>Hideki Tsunashima (AIST), Kosuke Arase (Mercari, Inc.), Antony Lam (Mercari, Inc.), Hirokatsu Kataoka (AIST) <br>
>Sensors 2020

## First doing
1. Please use the `recursive` option so that cloning the git submodule when you clone this repository.
```
git clone --recursive https://github.com/maguro27/UVIRT.git
```
2. Change the line 17, 360 and 362 of `gdwct/run.py`.
```
--- a/test_run.py
+++ b/test_run.py
@@ -14,15 +14,7 @@ from scipy.linalg import block_diag

 class Run(object):
     def __init__(self, config):
-        self.data_loader = get_loader(
-            config["DATA_PATH"],
-            crop_size=config["CROP_SIZE"],
-            resize=config["RESIZE"],
-            batch_size=config["BATCH_SIZE"],
-            dataset=config["DATASET"],
-            mode=config["MODE"],
-            num_workers=config["NUM_WORKERS"],
-        )
+        self.data_loader = 0

         self.config = config
         self.device = torch.device(
@@ -477,6 +469,6 @@ def main():
         run.test()


-config = ges_Aonfig("configs/config.yaml")
+config = ges_Aonfig("configs/mpv.yaml")

-main()
\ No newline at end of file
+# main()
```

3. Move `__init__.py` to `gdwct`.
```
cd UVIRT
mv __init__.py gdwct
```


## Explanation of this repository
- `configs`: Including the config file.
- `datasets`: Dataset directory. It includes the train and test pair text files.
- `fid`: For computing FID. The modified version scripts of [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- `gdwct`: Git submodule. We use [the official PyTorch implementation of the GDWCT paper](https://github.com/WonwoongCho/GDWCT).
- `.gitmodules`: For git submodule.
- `__init__.py`: For initializing the gdwct directory.
- `data_loader_mpv.py`: Dataloader file for overwriting the gdwct dataloader.
- `googleDrive_download_folder.py`: Script using PyDrive for downloading the preprocessed MPV dataset.
- `make_pair_text.py`: The script of making a unpair text of the train and test datasets for computing FID.
- `requirements.txt`: For installing each modules.
- `run_uvirt.py`: Main running script.
- `save_img_mpv.py`: Saving function for run_uvirt.py.

## Dependencies
This repository environment is as follow. We did not check other environments.
- `Python`: 3.6
- `PyTorch`: 1.6.0
- `Torchvision`: 0.7.0
- `cuda toolkit`: 10.2.89
- `cuDNN`: 8.0.2
- `GPU`: NVIDIA Tesla V100 (16GB VRAM)

Other requirements are in `requirements.txt` and can be installed with
```
pip install -r requirements.txt
```

## Dataset
You can select the two types of downloading the preprocessed MPV dataset.
1. Manually download the dataset from GoogleDrive: [The preprocessed MPV dataset](https://drive.google.com/drive/folders/1oIpKLhc5Bwaz9IDobSGdk0UQRAuXZ-Wr?usp=sharing) <br>
Please add the dataset into the `datasets` directory, after you complete downloading the dataset. Afterthat,
```
cd UVIRT/datasets
unzip MPV_distributed.zip
mv MPV_distributed/* MPV_supervised
rm -r MPV_distributed
rm -r MPV_distributed.zip
```
2. Use `googleDrive_download_folder.py`. This script can skip a confirmation page in GoogleDrive when downloading.
If you want to use the below script, please set up PyDrive.
See [PyDrive document](https://pythonhosted.org/PyDrive/).
```
cd UVIRT
python googleDrive_download_folder.py -p 1oIpKLhc5Bwaz9IDobSGdk0UQRAuXZ-Wr -s ./datasets
cd datasets
unzip MPV_distributed.zip
mv MPV_distributed/* MPV_supervised
rm -r MPV_distributed
rm -r MPV_distributed.zip
```

## Test visualiation with pre-trained models on the MPV dataset
You can visualize virtual try-on results on the test set in the two steps.
1. Download the pre-trained models as well as downloading the dataset.
- Manually download the pre-trained models from GoogleDrive: [The pre-trained models](https://drive.google.com/drive/folders/1IZBOZX-vUxy3cI-SGmJ3mRgFHzI3DpKg?usp=sharing) <br>
Please make the `./gdwct/models` directory and add the dataset into the `./gdwct/models` directory, after you complete downloading the pre-trained models. Afterthat,
```
unzip ${downloaded_zip_file_name}
cd uvirt_pretrained_models
mv ./* ../
cd ..
rm -r uvirt_pretrained_models
rm -r ${downloaded_zip_file_name}
```
- Use `googleDrive_download_folder.py`.
```
cd UVIRT
python googleDrive_download_folder.py -p 1IZBOZX-vUxy3cI-SGmJ3mRgFHzI3DpKg -s ./gdwct/models/
```
2. Run the script.
```
python run_uvirt.py -m test -l -s 490000 --model_save_path gdwct/models
```

## Training
```
python run_uvirt.py
```

## Evaluation
You can evaluate the FID of our model trained on the MPV dataset in the three steps.
1. Make random pairs of the test dataset.
```
python make_pair_text.py
```
2. Generate try-on images.
```
python run_uvirt.py -m test -l -s 490000 --model_save_path gdwct/models -f -b 1
```
3. Evaluate the FID.
```
python ./fid/fid_score_iterable.py ./datasets/MPV_supervised/test/image ./test_results -c 0 -r mpv_results_mpv_192_256
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
This repository use the official implementation of the paper "Image-to-Image Translation via Group-wise Deep Whitening-and-Coloring Transformation". This repository structure is inspired by [SPACE](https://github.com/zhixuan-lin/SPACE). We thank these repositories for help with the code release.
