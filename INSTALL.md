# Installation

## Requirements
```bash
    - python >= 3.8
    - pytorch >= 1.7.0
    - torchvision        
    - timm == 0.4.12      
    - scipy     
    - pandas
    - wandb    
    - pytorch_metric_learning 
```

## Setup
First, clone the repository:
```bash
git clone https://github.com/tjddus9597/HIER-CVPR23
cd HIER-CVPR23
```
Either install the requirements listed above manually, or use our conda environment:
```bash
conda env create --file environment.yml
conda activate hier
```

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.12
pip install scipy
pip install wandb
pip install pytorch-metric-learning
pip install pandas
```