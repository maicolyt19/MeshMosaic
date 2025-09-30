<p align="center">
  <img src="assets/title.png" alt="MeshMosaic">
</p>

<div align="center">

# Scaling Artist Mesh Generation via Local-to-Global Assembly

</div>

<div align="center">

[Rui Xu](https://ruixu.me/)<sup>1</sup>, [Tianyang Xue](https://xty.im/)<sup>1</sup>, [Qiujie Dong](https://qiujiedong.github.io/)<sup>1</sup>, [Le Wan](#)<sup>2</sup>, [Zhe Zhu](https://scholar.google.com/citations?user=pM4ebg0AAAAJ&hl=en)<sup>2</sup>, [Peng Li](https://penghtyx.github.io/yuki-lipeng)<sup>3</sup>, [Zhiyang Dou](https://frank-zy-dou.github.io/)<sup>1</sup>, [Cheng Lin](https://clinplayer.github.io/)<sup>4</sup>, [Shiqing Xin](https://irc.cs.sdu.edu.cn/~shiqing/index.html)<sup>5</sup>, [Yuan Liu](https://liuyuan-pal.github.io/)<sup>3</sup>, [Wenping Wang](https://engineering.tamu.edu/cse/profiles/Wang-Wenping.html)<sup>6</sup>, [Taku Komura](https://www.cs.hku.hk/index.php/people/academic-staff/taku)<sup>1</sup>

**Affiliations:**
<sup>1</sup> The University of Hong Kong
<sup>2</sup> Tencent Visvise  
<sup>3</sup> Hong Kong University of Science and Technology
<sup>4</sup> Macau University of Science and Technology
<sup>5</sup> Shandong University
<sup>6</sup> Texas A&M University


</div>

> 🚀 **Official code of MeshMosaic: Scaling Artist Mesh Generation via Local-to-Global Assembly**

<div align="center">
<a href='http://arxiv.org/abs/2509.19995'><img src='https://img.shields.io/badge/arXiv-2509.19995-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://xrvitd.github.io/MeshMosaic/index.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/Xrvitd/MeshMosaic"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://youtu.be/TO5CqY5UHvI'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'>
</div>

> 🚀 We are preparing the codebase for public release. Stay tuned!

## 📋 Release Todo List

- [x] Release pretrained checkpoints
- [x] Release inference code
- [ ] Release data preprocessing code
- [ ] Release training code

## Environment Setup

We tested on A100, A800 and H20 GPUs with CUDA 12.4 and CUDA 11.8. Follow the steps below to set up the environment.

### 1) Create Conda env and install PyTorch (CUDA 12.4)
```bash
conda create -n MeshMosaic python=3.12 -y
conda activate MeshMosaic

# PyTorch 2.5.1 + CUDA 12.4
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

# Core dependencies
pip install -U xformers==0.0.28.post3
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install packaging
```

### 2) Install FlashAttention and custom kernels
```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install

# If you encounter build issues, try FlashAttention==2.8.0 or 2.7.3

cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
```

### 3) Install remaining Python packages
```bash
pip install pymeshlab jaxtyping boto3 trimesh beartype lightning safetensors \
  open3d omegaconf sageattention triton scikit-image transformers gpustat \
  wandb pudb
pip install libigl
```

> Note: For CUDA 11.8 users, install the corresponding PyTorch/cu118 wheels and compatible `torch-cluster` build.

## Usage

- The script `sample.sh` demonstrates mesh generation. Input is an OBJ where each part is stored as a distinct connected component within a single file (see folder `input_pf`).
- Pre-trained weights are available on Hugging Face: [here](https://huggingface.co/Xrvitd/MeshMosaic).

Or run directly with the following command:
```bash
torchrun --nproc-per-node=1 --master_port=61107 sampleGPCBD.py \
  --model_path "ckpt/final.bin" \
  --steps 40000 \
  --input_path input_pf \
  --output_path output \
  --repeat_num 4 \
  --uid_list "" \
  --temperature 0.5
```

If you do not have connected-component inputs, you can use the code in the `PartField` folder to convert your own mesh into an OBJ with semantic segmentation. Below is a concise workflow.

### PartField environment setup

- Option A: Manual installation
```bash
conda create -n partfield python=3.10 -y
conda activate partfield

conda install -y nvidia/label/cuda-12.4.0::cuda

pip install psutil
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope \
  potpourri3d simple_parsing arrgh open3d
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

sudo apt-get update && sudo apt-get install -y libx11-6 libgl1 libxrender1
pip install vtk
```

- Option B: Use provided environment file
```bash
conda env create -f environment.yml
conda activate partfield
```

### Prepare checkpoint directory
```bash
mkdir -p model
```
Download the pretrained checkpoint: [Trained on Objaverse](https://huggingface.co/mikaelaangel/partfield-ckpt/blob/main/model_objaverse.ckpt).

Note: Due to licensing restrictions, the model also trained on PartNet cannot be released.

### Extract Feature Field
Run from the `PartField` project directory.
```bash
python partfield_inference.py -c configs/final/demo.yaml \
  --opts continue_ckpt model/model_objaverse.ckpt \
         result_name partfield_features/objaverse \
         dataset.data_path data/objaverse_samples
```

### Part segmentation
We use agglomerative clustering for mesh part segmentation.
```bash
python run_part_clustering.py \
  --root exp_results/partfield_features/objaverse \
  --dump_dir exp_results/clustering/objaverse \
  --source_dir data/objaverse_samples \
  --use_agglo True \
  --max_num_clusters 30 \
  --option 0
```

### Split connected components
```bash
python split_connected_component.py
```


Please follow our progress for updates. Training code and more resources will be released soon!

## Ack
Our code is based on these wonderful works:
* **[DeepMesh](https://github.com/zhaorw02/DeepMesh)**
* **[PartField](https://github.com/nv-tlabs/PartField)**
* **[BPT](https://github.com/Tencent-Hunyuan/bpt)**
* [Michelangelo](https://github.com/NeuralCarver/Michelangelo)






## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{xu2025meshmosaic,
  title={MeshMosaic: Scaling Artist Mesh Generation via Local-to-Global Assembly},
  author={Xu, Rui and Xue, Tianyang and Dong, Qiujie and Wan, Le and Zhu, Zhe and Li, Peng and Dou, Zhiyang and Lin, Cheng and Xin, Shiqing and Liu, Yuan and others},
  journal={arXiv preprint arXiv:2509.19995},
  year={2025}
}
```
