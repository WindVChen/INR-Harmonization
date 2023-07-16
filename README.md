<div align="center">

<h1><a href="https://arxiv.org/abs/2303.01681">Dense Pixel-to-Pixel Harmonization via Continuous Image Representation</a></h1>

**[Jianqi Chen](https://windvchen.github.io/), [Yilan Zhang](https://scholar.google.com.hk/citations?hl=en&user=wZ4M4ecAAAAJ), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), [Keyan Chen](https://scholar.google.com.hk/citations?hl=en&user=5RF4ia8AAAAJ), and [Zhenwei Shi](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**

![](https://komarev.com/ghpvc/?username=windvchenINR-Harmonization&label=visitors)
![GitHub stars](https://badgen.net/github/stars/windvchen/INR-Harmonization)
[![](https://img.shields.io/badge/license-Apache--2.0-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2303.01681-b31b1b.svg)](https://arxiv.org/abs/2303.01681)

</div>

### Share us a :star: if this repo does help

This repository is the official implementation of *HINet*. If you encounter any question, please feel free to contact us. You can create an issue or just send email to me windvchen@gmail.com. Also welcome for any idea exchange and discussion.

## Updates

[**07/16/2023**] Code is initially public.

[**03/06/2023**] Source code and pretrained models will be publicly accessible.

## TODO
- [x] Initial code release.
- [ ] Add pretrained model weights.
- [ ] Add the efficient splitting strategy for inferencing on original resolution images.

## Table of Contents

- [Abstract](#Abstract)
- [Requirements](#Requirements)
- [Training](#Training)
  - [Train in low resolution (LR) mode](#Train-in-low-resolution-(LR)-mode)
  - [Train in high resolution (HR) mode](#Train-in-high-resolution-(HR)-mode-(E.g,-2048x2048))
  - [Train in original resolution mode](#Train-in-original-resolution-mode)
- [Evaluation](#Evaluation)
  - [Inference in low resolution (LR) mode](#Inference-in-low-resolution-(LR)-mode)
  - [Inference in high resolution (HR) mode](#Inference-in-high-resolution-(HR)-mode-(E.g,-2048x2048))
  - [Inference in original resolution mode](#Inference-in-original-resolution-mode)
- [Results](#Results)
- [Citation & Acknowledgments](#Citation-&-Acknowledgments)
- [License](#License)


## Abstract

![DiffAttack's framework](assets/network.png)

High-resolution (HR) image harmonization is of great significance in real-world applications such as image synthesis and image editing. However, due to the high memory costs, existing dense pixel-to-pixel harmonization methods are mainly focusing on processing low-resolution (LR) images. Some recent works resort to combining with color-to-color transformations but are either limited to certain resolutions or heavily depend on hand-crafted image filters. In this work, we explore leveraging the implicit neural representation (INR) and propose a novel ***image Harmonization method based on Implicit neural Networks (HINet)***, which to the best of our knowledge, is ***the first dense pixel-to-pixel method applicable to HR images without any hand-crafted filter design***.  Inspired by the Retinex theory, we decouple the MLPs into two parts to respectively capture the content and environment of composite images. A Low-Resolution Image Prior (LRIP) network is designed to alleviate the Boundary Inconsistency problem, and we also propose new designs for the training and inference process. Extensive experiments have demonstrated the effectiveness of our method compared with state-of-the-art methods. Furthermore, some interesting and practical applications of the proposed method are explored.

## Requirements

1. Software Requirements
    - Python: 3.8
    - CUDA: 11.3
    - cuDNN: 8.4.1

   To install other requirements:

   ```
   pip install -r requirements.txt
   ```

2. Datasets
   - We train and evaluate on the [iHarmony4 dataset](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4). Please download the dataset in advance, and arrange them into the following structure:

   ```
   ├── dataset_path
      ├── HAdobe5k
         ├── composite_images
         ├── masks
         ├── real_images
      ├── HCOCO
      ├── Hday2night
      ├── HFlickr
      IHD_test.txt
      IHD_train.txt
   ```

   - Before training we resize HAdobe5k subdataset so that each side is smaller than 2048. This is for quick data loading. The resizing script can refer to [resize_hdataset.ipynb](https://github.com/SamsungLabs/image_harmonization/blob/master/notebooks/resize_hdataset.ipynb).
   
   - For training or evaluating on the original resolution of iHarmony4 dataset. Please newly create a `HAdobe5kori` directory with the original HAdobe5k images in it.

3. Pre-trained Models
   - We adopt [HRNetV2](https://github.com/HRNet/HRNet-Image-Classification) as our encoder, you can download the weight from [here](https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&id=F7FD0B7F26543CEB%21112&cid=F7FD0B7F26543CEB&parId=root&parQt=sharedby&parCid=C8304F01C1A85932&o=OneUp) and save the weight in `pretrained_models` directory.
   - In the following table, we provide some model weights pretrained under different resolutions: 

## Training

### Train in low resolution (LR) mode

```bash
python train.py --dataset_path {dataset_path} --base_size 256 --input_size 256 --INR_input_size 256 --hr_train False --isFullRes False
```
- `dataset_path`: the path of the iHarmony4 dataset.
- `base_size`: the size of the input image to encoder.
- `input_size`: the size of the target resolution.
- `INR_input_size`: the size of the input image to the INR decoder.
- `hr_train`: whether to train in high resolution (HR) mode, i.e., using RSC strategy (See Section 3.4 in the paper).
- `isFullRes`: whether to train in full/original resolution mode.

- (More parameters' information could be found in codes ...)

### Train in high resolution (HR) mode (E.g, 2048x2048)

If not use RSC strategy, the training command is as follows:
```bash
python train.py --dataset_path {dataset_path} --base_size 256 --input_size 2048 --INR_input_size 2048 --hr_train False --isFullRes False
```

If use RSC strategy, the training command is as follows:
```bash
python train.py --dataset_path {dataset_path} --base_size 256 --input_size 2048 --INR_input_size 2048 --hr_train True --isFullRes False
```

### Train in original resolution mode
```bash
python train.py --dataset_path {dataset_path} --base_size 256 --hr_train True --isFullRes True
```

## Evaluation

### Evaluation in low resolution (LR) mode

```bash
python inference.py --dataset_path {dataset_path} --base_size 256 --input_size 256 --INR_input_size 256 --hr_train False --isFullRes False
```

### Evaluation in high resolution (HR) mode (E.g, 2048x2048)

```bash
python inference.py --dataset_path {dataset_path} --base_size 256 --input_size 2048 --INR_input_size 2048 --isFullRes False
```

### Train in original resolution mode
```bash
python inference.py --dataset_path {dataset_path} --base_size 256 --hr_train True --isFullRes True
```

## Results

![Metrics](assets/metrics.png#pic_center)
![Visual comparisons](assets/visualizations.png#pic_center)
![Visual comparisons2](assets/visualizations2.png#pic_center)


## Citation & Acknowledgments
If you find this paper useful in your research, please consider citing:
```
@article{chen2023dense,
  title={Dense Pixel-to-Pixel Harmonization via Continuous Image Representation},
  author={Chen, Jianqi and Zhang, Yilan and Zou, Zhengxia and Chen, Keyan and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2303.01681},
  year={2023}
}
```

## License
This project is licensed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.