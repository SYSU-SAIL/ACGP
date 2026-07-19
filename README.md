<h1 align="center">Beyond Context Bias: Adaptive Instance
Placement for Robust 3D Instance Segmentation</h1>

<p align="center">
  <a href="#">Paper</a>
  ·
  <a href="https://yangrongkun.github.io/">Project Page</a>
  ·
  <a href="#citation">BibTeX</a>
</p>

<p align="center">
  This repository contains the official implementation of Adaptive Instance Placement for Robust 3D Instance Segmentation (ACGP).
</p>

<p align="center">
  <img src="docs/images/framework.png" alt="main_figure" width="900" />
</p>

<p align="center">
  <strong>Overall framework of ACGP.</strong> Given a 3D scene and an instance database, ACGP selects instances via category-balanced sampling, queries their category-conditioned placement types, and places them at geometrically valid locations using occupancy- and support-aware validation to generate augmented scenes. <strong>The pseudo-code</strong> on the right summarizes the complete pipeline of instance-level augmented scene generation.
</p>

<p align="center">
  The core ACGP instance placement implementation can be found in
  <a href="pointcept/datasets/instance_augmentor_occupancy.py"><code>pointcept/datasets/instance_augmentor_occupancy.py</code></a>.
</p>

## 📢 News


<details>
<summary><b>Update:  ACGP achieves the SOTA performance on ScanNet200 test set for 3D instance segmentation benchmark. Test scores accessed on 12 July, 2026. <a href="https://kaldir.vc.in.tum.de/scannet_benchmark/scannet200_semantic_instance_3d" target="_blank">ScanNet200 test set</a> </b> (The results are provided by official based on this repo)</summary>

![image](./docs/images/scannet200_test_benchmark.png)
</details>


<details>
<summary><b>ACGP achieves the SOTA performance on ScanNet++ V2 test set for 3D instance segmentation. Test scores accessed on 10 July, 2026. <a href="https://kaldir.vc.in.tum.de/scannetpp/benchmark/insseg" target="_blank">ScanNet++ V2 test set</a></b> </summary>

![image](./docs/images/scannetpp_benchmark.png)
</details>


## :floppy_disk: ACGP Trained Results
<table>
  <thead>
    <tr>
      <th align="center">Benchmark</th><th align="center">Model</th><th align="center">mAP</th><th align="center">AP50</th><th align="center">AP25</th><th align="center">Tensorboard</th><th align="center">Exp Record</th><th align="center">Model</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="2" align="center" valign="middle">ScanNet++ Val</td><td align="center">Volt-SPFormer</td><td align="center">37.2</td><td align="center">56.3</td><td align="center">64.6</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet%2B%2B/Volt-SPFormer/events.out.tfevents.1780793811">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet%2B%2B/Volt-SPFormer/train.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet%2B%2B/Volt-SPFormer/Volt_SPFormer_ScanNetpp.pth">Link</a></td></tr>
    <tr><td align="center">SGIFormer</td><td align="center">35.4</td><td align="center">52.4</td><td align="center">61.4</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet%2B%2B/SGIFormer/events.out.tfevents.1765376744">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet%2B%2B/SGIFormer/train.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet%2B%2B/SGIFormer/sgiformer_scannetpp.pth">Link</a></td></tr>
    <tr><td rowspan="4" align="center" valign="middle">ScanNet200 Val</td><td align="center">Volt-SPFormer</td><td align="center">39.6</td><td align="center">50.2</td><td align="center">55.2</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/Volt-SPFormer/events.out.tfevents.1780750167">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/Volt-SPFormer/train.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/Volt-SPFormer/Volt_SPFormer_ScanNet200.pth">Link</a></td></tr>
    <tr><td align="center">SGIFormer-L</td><td align="center">32.9</td><td align="center">43.3</td><td align="center">49.3</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/SGIFormer/events.out.tfevents.1778374769">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/SGIFormer/train.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/SGIFormer/sgiformer_scannet200.pth">Link</a></td></tr>
    <tr><td align="center">ISBNet</td><td align="center">26.2</td><td align="center">34.4</td><td align="center">39.2</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/ISBNet/events.out.tfevents.1763898474">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/ISBNet/20251123_194754.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/ISBNet/isbnet_scannet200.pth">Link</a></td></tr>
    <tr><td align="center">TD3D</td><td align="center">24.2</td><td align="center">36.2</td><td align="center">42.2</td><td align="center">-</td><td align="center">-</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannet200/TD3D/td3d_scannet200.pth">Link</a></td></tr>
    <tr><td rowspan="4" align="center" valign="middle">ScanNet Val</td><td align="center">SGIFormer-L</td><td align="center">61.6</td><td align="center">81.6</td><td align="center">89.2</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/SGIFormer/events.out.tfevents.1774235401">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/SGIFormer/train.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/SGIFormer/sgiformer_scannetv2.pth">Link</a></td></tr>
    <tr><td align="center">ISBNet</td><td align="center">56.1</td><td align="center">73.6</td><td align="center">81.8</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/ISBNet/events.out.tfevents.1781668966">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/ISBNet/20260617_120246.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/ISBNet/isbnet_scannetv2.pth">Link</a></td></tr>
    <tr><td align="center">SoftGroup</td><td align="center">44.0</td><td align="center">65.3</td><td align="center">77.9</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/SoftGroup/events.out.tfevents.1764642550">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/SoftGroup/20251202_102910.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/SoftGroup/softgroup_scannetv2.pth">Link</a></td></tr>
    <tr><td align="center">TD3D</td><td align="center">47.6</td><td align="center">71.3</td><td align="center">82.1</td><td align="center">-</td><td align="center">-</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/scannetv2/TD3D/td3d_scannetv2.pth">Link</a></td></tr>
    <tr><td rowspan="3" align="center" valign="middle">S3DIS Area5</td><td align="center">ISBNet</td><td align="center">58.2</td><td align="center">71.3</td><td align="center">77.7</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/s3dis/ISBNet/events.out.tfevents.1763992796">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/s3dis/ISBNet/train.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/s3dis/ISBNet/isbnet_s3dis.pth">Link</a></td></tr>
    <tr><td align="center">SoftGroup</td><td align="center">53.5</td><td align="center">66.7</td><td align="center">73.7</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/s3dis/SoftGroup/events.out.tfevents.1764510534">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/s3dis/SoftGroup/20251130_214854.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/s3dis/SoftGroup/softgroup_s3dis.pth">Link</a></td></tr>
    <tr><td align="center">TD3D</td><td align="center">53.1</td><td align="center">68.2</td><td align="center">75.9</td><td align="center">-</td><td align="center">-</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/s3dis/TD3D/td3d_s3dis.pth">Link</a></td></tr>
    <tr><td align="center">STPLS3D Val</td><td align="center">ISBNet</td><td align="center">52.1</td><td align="center">68.9</td><td align="center">76.0</td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/stpls3d/ISBNet/events.out.tfevents.1764129772">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/stpls3d/ISBNet/20251126_120252.log">Link</a></td><td align="center"><a href="https://huggingface.co/yangrk/ACGP/blob/main/stpls3d/ISBNet/isbnet_stpls3d.pth">Link</a></td></tr>
  </tbody>
</table>



## Setup

This repository is built on top of [Pointcept](https://github.com/Pointcept/Pointcept/blob/04a0232b70f5c7091ffdc6bfe7a476e3eb7daff2) and incorporates components from [SGIFormer](https://github.com/RayYoh/SGIFormer/blob/4c05d57bbbd676b6a2398b03deac916e603a9dd7) and [Volt](https://github.com/YilmazKadir/Volt) for instance segmentation. 

### Dependencies
We recommend using [`uv`](https://docs.astral.sh/uv/#highlights), a fast Python package and environment manager, to install the environment.

To install `uv` on macOS and Linux, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the environment with:
```bash
# Make sure to load CUDA 12.6 beforehand
# This will automatically create a virtual environment (.venv) and install dependencies from pyproject.toml
uv sync
source .venv/bin/activate
```

## Data Preprocessing
Follow the dataset setup instructions in the [Pointcept README](https://github.com/Pointcept/Pointcept/blob/04a0232b70f5c7091ffdc6bfe7a476e3eb7daff2/README.md).

### Indoor Datasets
Preprocessing for indoor datasets is identical to Pointcept.

### Outdoor Datasets
Preprocessing for outdoor datasets is identical to ISBNet.




### Instance Segmentation

First, run the preprocessing script to generate superpoints for ScanNet and ScanNet200.
```bash
python pointcept/datasets/preprocessing/scannet/preprocess_superpoints.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```

Download the pretrained Volt-S backbone weights from [HuggingFace](https://huggingface.co/KadirYilmaz/Volt/tree/main)
```bash
mkdir -p weights
curl -L -o weights/volt-small-scannet.pth https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_small/scannet/model/model_last.pth
curl -L -o weights/volt-small-scannet200.pth https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_small/scannet200/model/model_last.pth
```
Alternatively you can train them yourself using the corresponding configs above.

Then, run the training script with the `insseg-spformer-volt-S-0-base` config for scannet/scannet200

```bash
### ScanNet
sh scripts/train.sh -g 4 -d scannet -c insseg-spformer-volt-S-0-base -n insseg-volt
### ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c insseg-spformer-volt-S-0-base -n insseg-volt
```

<!-- ## Model Zoo

We provide the experiment directories, including configs, logs, and checkpoints. The experiments can also be seen from [Hugging Face](#).

### 3D Instance Segmentation: Baseline Training

| Model | Dataset | Val mAP | Exp. Dir |
| :--- | :--- | :---: | :---: |
| Volt-S | ScanNet | 76.3 | [link](#) |
| Volt-S | ScanNet200 | 36.1 | [link](#) |
| Volt-S | ScanNet++ | 50.2 | [link](#) |


### 3D Instance  Segmentation: ACGP Training

| Model | Dataset | Val mAP | Exp. Dir |
| :--- | :--- | :---: | :---: |
| Volt-S | ScanNet | 80.2 | [link](#) |
| Volt-S | ScanNet200 | 38.5 | [link](#) |
| Volt-S | ScanNet++ | 50.2 | [link](#) | -->

## Citation

If you use our work in your research, please use the following BibTeX entry.

```
@misc{yang2026acgp,
  title  = {Beyond Context Bias: Adaptive Instance Placement for Robust 3D Instance Segmentation},
  author = {Rongkun Yang, Ye Zhang, Longguang Wang, Zhiheng Fu, Lian Xu, Yulan Guo},
  year   = {2026},
  url    = {https://github.com/SYSU-SAIL/ACGP}
}

```



## Acknowledgements
Code is built based on [Volt](https://github.com/YilmazKadir/Volt), [PointCept](https://github.com/Pointcept/Pointcept), and [SGIFormer](https://github.com/RayYoh/SGIFormer). We sincerely thank the authors for sharing their code.


