# Out-of-Context Misinformation Detection via Variational Domain-Invariant Learning with Test-Time Training (AAAI 2026)
Xi Yang, Han Zhang, Zhijian Lin, Yibiao Hu, and Hong Han*

This repository contains code of our paper "Out-of-Context Misinformation Detection via Variational Domain-Invariant Learning with Test-Time Training", accepted by the AAAI Conference on Artificial Intelligence (AAAI) 2026.

## Paper
Arxiv Link: [https://arxiv.org/pdf/2511.10213](https://arxiv.org/pdf/2511.10213)

## Downloading the dataset
You can find the NewsCLIPpings dataset following the instructions in [this](https://github.com/S-Abdelnabi/OoC-multi-modal-fc?tab=readme-ov-file) repository.

## Requirements
The required packages are listed in ``requirements_py38.txt``. The setup is based on the environment used for [ConDA-TTT](https://github.com/gymbeijing/ooc-detection-2/blob/master/README.md).

You can install the dependencies using pip:
```
conda create -n venv_py38 python=3.8 -y
conda activate venv_py38
pip install -r requirements_py38.txt
```

## Running
For NewsCLIPpings:
```
(venv_py38) python -m trainers.train_VDT --batch_size 256 --max_epochs 20 --target_domain bbc,guardian --base_model blip-2 --loss_type simclr
```

## Data Processing
You can processe the NewsCLIPpings dataset with `` src/save_all_tensor_to_one_file_newsclippings.py``.

## Acknowledgements
This codebase is developed based on the implementation of [ConDA-TTT](https://github.com/gymbeijing/ooc-detection-2/blob/master/README.md). 
We sincerely thank the original authors for sharing their code and groundwork.

## Citation
If you find this work useful for your research, please consider citing our paper:
```
@misc{yang2025outofcontextmisinformationdetectionvariational,
      title={Out-of-Context Misinformation Detection via Variational Domain-Invariant Learning with Test-Time Training}, 
      author={Xi Yang and Han Zhang and Zhijian Lin and Yibiao Hu and Hong Han},
      year={2025},
      eprint={2511.10213},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.10213}, 
}
```
