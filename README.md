<div align="center">

<h1><a href="https://arxiv.org/abs/2303.01681">Dense Pixel-to-Pixel Harmonization via <br /> Continuous Image Representation</a></h1>

**[Jianqi Chen](https://windvchen.github.io/), [Yilan Zhang](https://scholar.google.com.hk/citations?hl=en&user=wZ4M4ecAAAAJ), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), [Keyan Chen](https://scholar.google.com.hk/citations?hl=en&user=5RF4ia8AAAAJ), and [Zhenwei Shi](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**

![](https://komarev.com/ghpvc/?username=windvchenINR-Harmonization&label=visitors)
![GitHub stars](https://badgen.net/github/stars/windvchen/INR-Harmonization)
[![](https://img.shields.io/badge/license-Apache--2.0-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2303.01681-b31b1b.svg)](https://arxiv.org/abs/2303.01681)

</div>

### Share us a :star: if this repo does help

This repository is the official implementation of *HINet*. If you encounter any question, please feel free to contact us. You can create an issue or just send email to me windvchen@gmail.com. Also welcome for any idea exchange and discussion.

## Updates

[**07/18/2023**] Check out our new work [***Diff-Harmonization***](https://github.com/WindVChen/Diff-Harmonization), which is a **Zero-Shot Harmonization** method based on *Diffusion Models*!😊

[**07/17/2023**] Pretrained weights have been released. Feel free to try that!👋👋

[**07/16/2023**] The code is initially public. 🥳

[**03/06/2023**] Source code and pretrained models will be publicly accessible.

## TODO
- [x] Initial code release.
- [x] Add pretrained model weights.
- [ ] Add the efficient splitting strategy for inferencing on original resolution images.
- [ ] Add Gradio demo.

## Table of Contents

- [Abstract](#abstract)
- [Requirements](#requirements)
- [Training](#training)
  - [Train in low resolution (LR) mode](#train-in-low-resolution--lr--mode)
  - [Train in high resolution (HR) mode](#train-in-high-resolution--hr--mode--eg-2048x2048-)
  - [Train in original resolution mode](#train-in-original-resolution-mode)
- [Evaluation](#evaluation)
  - [Evaluation in low resolution (LR) mode](#evaluation-in-low-resolution--lr--mode)
  - [Evaluation in high resolution (HR) mode](#evaluation-in-high-resolution--hr--mode--eg-2048x2048-)
  - [Evaluation in original resolution mode](#evaluation-in-original-resolution-mode)
- [Results](#results)
- [Citation & Acknowledgments](#citation--acknowledgments)
- [License](#license)


## Abstract

![HINet's framework](assets/network.png)

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

   - Before training we resize HAdobe5k subdataset so that each side is smaller than 1024. This is for quick data loading. The resizing script can refer to [resize_Adobe.py](tools/resize_Adobe.py).
   
   - For training or evaluating on the original resolution of iHarmony4 dataset. Please newly create a `HAdobe5kori` directory with the original HAdobe5k images in it.

   - If you want to train and evaluate only on HAdobe5k subdataset (see Table 1 in the paper), you can modify the `IHD_train.txt` and `IHD_test.txt` in [train.py](train.py) to only contain the HAdobe5k images.

3. Pre-trained Models
   - We adopt [HRNetV2](https://github.com/HRNet/HRNet-Image-Classification) as our encoder, you can download the weight from [here](https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&id=F7FD0B7F26543CEB%21112&cid=F7FD0B7F26543CEB&parId=root&parQt=sharedby&parCid=C8304F01C1A85932&o=OneUp) and save the weight in `pretrained_models` directory.
   - In the following table, we provide several model weights pretrained under different resolutions (Correspond to Table 1 in the paper): 

|                      Download Link                       |                         Model Descriptions                          |
|:--------------------------------------------------------:|:-------------------------------------------------------------------:|
| [Resolution_RAW_iHarmony4.pth][Resolution_RAW_iHarmony4] |  Train by RSC strategy with original resolution iHarmony4 dataset   |
| [Resolution_256_iHarmony4.pth][Resolution_256_iHarmony4] |           Train with 256*256 resolution iHarmony4 dataset           |
|  [Resolution_RAW_HAdobe5K.pth][Resolution_RAW_HAdobe5K]  | Train by RSC strategy with original resolution HAdobe5k subdataset  |
| [Resolution_2048_HAdobe5K.pth][Resolution_2048_HAdobe5K] | Train by RSC strategy with 2048*2048 resolution HAdobe5k subdataset |
| [Resolution_1024_HAdobe5K.pth][Resolution_1024_HAdobe5K] | Train by RSC strategy with 1024*1024 resolution HAdobe5k subdataset |

[Resolution_RAW_iHarmony4]: https://drive.google.com/file/d/1O9faWNk54mIzMaGZ1tmgm0EJpH20a-Fl/view?usp=drive_link
[Resolution_256_iHarmony4]: https://drive.google.com/file/d/1xym96LTP9a75UseDWGW2KRN1gyl3HPyM/view?usp=sharing
[Resolution_RAW_HAdobe5K]: https://drive.google.com/file/d/1JeUS5inuOM0pASKfu-tK9K7E5pGkP570/view?usp=drive_link
[Resolution_2048_HAdobe5K]: https://drive.google.com/file/d/18RxTfZsPEoi6kSS_UVEsUBYRBHAl4MfB/view?usp=drive_link
[Resolution_1024_HAdobe5K]: https://drive.google.com/file/d/1cOY74mN8gIz66watyoobZ1knrigkQyb5/view?usp=sharing

## Training

The intermediate output (including checkpoint, visualization, log.txt) will be saved in directory `logs/exp`.

### Train in low resolution (LR) mode

```bash
python train.py --dataset_path {dataset_path} --base_size 256 --input_size 256 --INR_input_size 256
```
- `dataset_path`: the path of the iHarmony4 dataset.
- `base_size`: the size of the input image to encoder.
- `input_size`: the size of the target resolution.
- `INR_input_size`: the size of the input image to the INR decoder.
- `hr_train`: whether to train in high resolution (HR) mode, i.e., using RSC strategy (See Section 3.4 in the paper).
- `isFullRes`: whether to train in full/original resolution mode.

- (More parameters' information could be found in codes ...)

### Train in high resolution (HR) mode (E.g, 2048x2048)

If **not use RSC strategy**, the training command is as follows: (For a single RTX 3090, it will lead to out-of-memory even `batch_size` is set to 2.)
```bash
python train.py --dataset_path {dataset_path} --base_size 256 --input_size 2048 --INR_input_size 2048
```

If **use RSC strategy**, the training command is as follows: (For a single RTX 3090, `batch_size` can set up to 6.)
```bash
python train.py --dataset_path {dataset_path} --base_size 256 --input_size 2048 --INR_input_size 2048 --hr_train
```

### Train in original resolution mode
```bash
python train.py --dataset_path {dataset_path} --base_size 256 --hr_train --isFullRes
```

## Evaluation

The intermediate output (including visualizations, log.txt) will be saved in directory `logs/test`.

**Notice:** Due to the resolution-agnostic characteristic of INR, you can inference images at any resolution not matter which resolution the model is trained on. Please refer to Table 4 and Table 5 in the paper. 

### Evaluation in low resolution (LR) mode

```bash
python inference.py --dataset_path {dataset_path} --pretrained {pretrained_weight} --base_size 256 --input_size 256 --INR_input_size 256
```

### Evaluation in high resolution (HR) mode (E.g, 2048x2048)

```bash
python inference.py --dataset_path {dataset_path} --pretrained {pretrained_weight} --base_size 256 --input_size 2048 --INR_input_size 2048
```

### Evaluation in original resolution mode
```bash
python inference.py --dataset_path {dataset_path} --pretrained {pretrained_weight} --base_size 256 --hr_train --isFullRes
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