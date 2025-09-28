<p align="center">
  <img src="assets/title.png" alt="MeshMosaic" width="400">
</p>

<div align="center">

# 🎨 MeshMosaic: Scaling Artist Mesh Generation via Local-to-Global Assembly

</div>

<div align="center">

**Authors:** [Rui Xu](https://ruixu.me/)<sup>1</sup>, [Tianyang Xue](https://xty.im/)<sup>1</sup>, [Qiujie Dong](https://qiujiedong.github.io/)<sup>1</sup>, Le Wan<sup>2</sup>, [Zhe Zhu](https://scholar.google.com/citations?user=pM4ebg0AAAAJ&hl=en)<sup>2</sup>, [Peng Li](https://penghtyx.github.io/yuki-lipeng)<sup>3</sup>, [Zhiyang Dou](https://frank-zy-dou.github.io/)<sup>1</sup>, [Cheng Lin](https://clinplayer.github.io/)<sup>4</sup>, [Shiqing Xin](https://irc.cs.sdu.edu.cn/~shiqing/index.html)<sup>5</sup>, [Yuan Liu](https://liuyuan-pal.github.io/)<sup>3†</sup>, [Wenping Wang](https://engineering.tamu.edu/cse/profiles/Wang-Wenping.html)<sup>6</sup>, [Taku Komura](https://www.cs.hku.hk/index.php/people/academic-staff/taku)<sup>1†</sup>

**Affiliations:**
- <sup>1</sup> The University of Hong Kong
- <sup>2</sup> Tencent Visvise  
- <sup>3</sup> Hong Kong University of Science and Technology
- <sup>4</sup> Macau University of Science and Technology
- <sup>5</sup> Shandong University
- <sup>6</sup> Texas A&M University

<sup>†</sup> Corresponding authors

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
- [ ] Release inference code
- [ ] Release data preprocessing code
- [ ] Release training code

Follow our progress for updates. Code and resources will be released soon!

## 📖 Abstract

Scaling artist-designed meshes to high triangle numbers remains challenging for autoregressive generative models. Existing transformer-based methods suffer from long-sequence bottlenecks and limited quantization resolution, primarily due to the large number of tokens required and constrained quantization granularity. These issues prevent faithful reproduction of fine geometric details and structured density patterns.

We introduce MeshMosaic, a novel local-to-global framework for artist mesh generation that scales to over 100K triangles—substantially surpassing prior methods, which typically handle only around 8K faces. MeshMosaic first segments shapes into patches, generating each patch autoregressively and leveraging shared boundary conditions to promote coherence, symmetry, and seamless connectivity between neighboring regions.

This strategy enhances scalability to high-resolution meshes by quantizing patches individually, resulting in more symmetrical and organized mesh density and structure. Extensive experiments across multiple public datasets demonstrate that MeshMosaic significantly outperforms state-of-the-art methods in both geometric fidelity and user preference, supporting superior detail representation and practical mesh generation for real-world applications.
