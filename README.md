# LlavaGuard
*LLAVAGUARD: VLM-based Safeguard for Vision Dataset Curation and Safety Assessment*

This is the official repository for [LlavaGuard](https://arxiv.org/abs/2406.05113), a versatile framework for evaluating visual content safety compliance. LlavaGuard is designed for both dataset annotation and safeguarding generative models.

📄 [Project Page](https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html)  
🤗 [Hugging Face Models](https://huggingface.co/collections/AIML-TUDA/llavaguard-665b42e89803408ee8ec1086)

### Model Repositories
- [AIML-TUDA/LlavaGuard-Dataset](https://huggingface.co/AIML-TUDA/LlavaGuard-Dataset)
- [AIML-TUDA/LlavaGuard-v1.2-0.5B-OV](https://huggingface.co/AIML-TUDA/LlavaGuard-v1.2-0.5B-OV)
- [AIML-TUDA/LlavaGuard-v1.2-7B-OV](https://huggingface.co/AIML-TUDA/LlavaGuard-v1.2-7B-OV)

<div align="center">
  <img src="figs/llavaguard_pipe.png" width="600" alt="LlavaGuard Pipeline">
</div>

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Dataset Generation](#dataset-generation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Safety Taxonomy](#safety-taxonomy)
- [Methodology](#methodology)

## Overview
LlavaGuard comes with an open pipline for building safety datasets as well as open pre-trained weights for vision safeguarding. The models can be used for:
- Direct inference (via [SGLang](https://github.com/sgl-project/sglang) and Hugging Face Transformers)
- Fine-tuning via LoRA
- Full model training

------------------------------------------------------------
## Installation

### For Inference
```bash
# Option 1: Using SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
docker build -f docker/Dockerfile -t sglang .

# Option 2: Using Transformers
pip install transformers torch
```
------------------------------------------------------------
## Usage

### Inference
We provide two inference options:

1. **Via SGLang**
   - See example scripts in `scripts/inference/sglang.ipynb`
   - Requires SGLang installation

2. **Via Transformers**  
   - See example scripts in `scripts/inference/transformers.ipynb`
   - Uses standard Hugging Face pipeline

The model outputs include:
- Safety rating ("Safe" or "Unsafe")
- Safety category classification
- Detailed rationale for the assessment

------------------------------------------------------------

### Generating LlavaGuard dataset
We offer a pipeline to create new datasets based on a specified version. These versions are defined in `llavaguard_config.py`. You can also generate custom datasets by extending this file.

- To build new datasets you have to define local paths and dataset/model configurations in `llavaguard_config.py`.
- To generate rationale for new datasets, see `scripts/data/generate_rationales.sh`.
- To build new datasets, see `scripts/data/prepare_datasets.sh`.

------------------------------------------------------------


### Training LlavaGuard
Example scripts for training LlavaGuard are available in the `scripts/train` directory. Note that you will need to install additional dependencies based on the model you wish to tune:

LlavaGuard: [LLaVA](https://github.com/haotian-liu/LLaVA)
LlavaGuard-OV: [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
QwenGuard: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

------------------------------------------------------------

### Evaluating LlavaGuard
A script for evaluating LlavaGuard is provided in `scripts/eval.sh`. The evaluation supports different deployment options which you can define using the engine. We recommend using SGLang for deployment. Depending on your chosen engine, you may need to install the corresponding dependencies:

SGLang: [SGLang](https://github.com/sgl-project/sglang)
VLLM: [VLLM](https://github.com/vllm-project/vllm)
LMdeploy: [LMdeploy](https://github.com/InternLM/lmdeploy)

------------------------------------------------------------

## Safety Taxonomy

Our different taxonomies and augmentation techniques can be found in `llavaguard/taxonomy`.

------------------------------------------------------------

## Methodology

This paper introduces Llavaguard, a suite of VLM-based vision safeguards that address the critical need for reliable tools in the era of large-scale data and models. To this end, we establish a novel open framework, describing a customizable safety taxonomy, data preprocessing, augmentation, and training setup. For teaching a VLM safeguard on safety, we further create a multimodal safety dataset with high-quality human expert annotations, where each image is labeled with a safety rating, category and rationale. We also employ advanced augmentations to support context-specific assessments. The resulting Llavaguard models, ranging from 0.5B to 7B, serve as a versatile tool for evaluating the safety compliance of visual content against flexible policies. In comprehensive experiments, Llavaguard outperforms both state-of-the-art safeguards and VLMs in accuracy and in flexibly handling different policies. Additionally, we demonstrate Llavaguard's performance in two real-world applications: large-scale dataset annotation and moderation of text-to-image models. We make our entire framework publicly available, including the dataset and model weights.


<div align="center">
  <img src="figs/LlavaGuard%20Safety%20Assessments.png" width="800"  alt="">
</div>

------------------------------------------------------------


## Citation
If you use LlavaGuard for your research, please cite our paper:

```
@incollection{helff2024llavaguard, 
            crossref = { https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html }, 
            key = { Best Runner-Up Paper Award at NeurIPS RBFM 2024 }, 
            booktitle = { Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops and Working Notes of the NeurIPS 2024 Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models (RBFM) }, 
            year = { 2024 }, 
            author = { Lukas Helff and Felix Friedrich and Manuel Brack and Patrick Schramowski and Kristian Kersting }, 
            title = { LLAVAGUARD: VLM-based Safeguard for Vision Dataset Curation and Safety Assessment }
}
```

------------------------------------------------------------

This repository aims to facilitate research and development in visual content safety. For any questions, suggestions, or issues, please open an issue or submit a pull request.
