# GURO

This repository contains the implementations of GURO, BayesGURO, and all other algorithms used in the paper [Active Preference Learning for Ordering Items In- and Out-of-Sample](https://arxiv.org/abs/2405.03059), published in NeurIPS 2024.

## Running the code

```bash
$ pip install -r requirements.txt
$ python plots.py
```

Run plots.py to reproduce Figure 4 in the Appendix. The number of seeds can be reduced if the process is taking too much time. Do this by changing ```n_seeds```. The same implementations of the algorithms have been used when performing all other experiments. 

## Data

The data used in our experiments can be found below:

* [X-RayAge](https://www.kaggle.com/competitions/spr-x-ray-age) - Paulo Kuriki, Felipe Kitamura, Lilian Mallagoli
* [ImageClarity](https://dbgroup.cs.tsinghua.edu.cn/ligl/crowdtopk) - Xiaohang Zhang, Guoliang Li, and Jianhua Feng
* [WiscAds](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0ZRGEE) - David Carlson and Jacob M. Montgomery
* [IMDB-WIKI-SbS](https://github.com/Toloka/IMDB-WIKI-SbS) - Nikita Pavlichenko and Dmitry Ustalov

## Citation

```
@inproceedings{bergstrom_2024_active,
 author = {Bergstr\"{o}m, Herman and Carlsson, Emil and Dubhashi, Devdatt and Johansson, Fredrik D.},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {71962--71997},
 publisher = {Curran Associates, Inc.},
 title = {Active preference learning for ordering items in- and out-of-sample},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/8443219a991f068c34d9491ad68ffa94-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
