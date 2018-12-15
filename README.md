# AutoNER

**Check Our New NER Toolkit🚀🚀🚀**
- **Inference**:
  - **[LightNER](https://github.com/LiyuanLucasLiu/LightNER)**: inference w. models pre-trained / trained w. *any* following tools, *efficiently*. 
- **Training**:
  - **[LD-Net](https://github.com/LiyuanLucasLiu/LD-Net)**: train NER models w. efficient contextualized representations.
  - **[VanillaNER](https://github.com/LiyuanLucasLiu/Vanilla_NER)**: train vanilla NER models w. pre-trained embedding.
- **Distant Training**:
  - **[AutoNER](https://shangjingbo1226.github.io/AutoNER/)**: train NER models w.o. line-by-line annotations and get competitive performance.

--------------------------------

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/autoner/badge/?version=latest)](http://autoner.readthedocs.io/en/latest/?badge=latest)

**No line-by-line annotations**, AutoNER trains named entity taggers with distant supervision.

Details about AutoNER can be accessed at: [https://arxiv.org/abs/1809.03599](https://arxiv.org/abs/1809.03599)

- [Model notes](#model-notes)
- [Benchmarks](#benchmarks)
- [Training](#training)
	- [Required Inputs](#required-inputs)
	- [Dependencies](#dependencies)
	- [Command](#command)
- [Citation](#citation)

## Model Notes

![AutoNER-Framework](docs/AutoNER-Framework.png)

## Benchmarks

| Method | Precision | Recall | F1 |
| ------------- |-------------| -----| -----|
| Supervised Benchmark | 88.84 | 85.16 | **86.96** |
| Dictionary Match | 93.93 | 58.35 | 71.98 |
| Fuzzy-LSTM-CRF | 88.27 | 76.75 | 82.11 |
| AutoNER | 88.96 | 81.00 | **84.80** |

## Training

### Required Inputs

- **Tokenized Raw Texts**
  - Example: ```data/BC5CDR/raw_text.txt```
    - One token per line.
    - An empty line means the end of a sentence.
- **Two Dictionaries**
  - **Core Dictionary w/ Type Info**
    - Example: ```data/BC5CDR/dict_core.txt```
      - Two columns (i.e., Type, Tokenized Surface) per line.
      - Tab separated.
    - How to obtain?
      - From domain-specific dictionaries.
  - **Full Dictionary w/o Type Info**
    - Example: ```data/BC5CDR/dict_full.txt```
      - One tokenized high-quality phrases per line.
    - How to obtain? 
      - From domain-specific dictionaries.
      - Applying the high-quality phrase mining tool on domain-specific corpus.
        - [AutoPhrase](https://github.com/shangjingbo1226/AutoPhrase) 
- **Pre-trained word embeddings**
  - Train your own or download from the web.
  - The example run uses ```embedding/bio_embedding.txt```, which can be downloaded from [our group's server](http://dmserv4.cs.illinois.edu/bio_embedding.txt). For example, ```curl http://dmserv4.cs.illinois.edu/bio_embedding.txt -o embedding/bio_embedding.txt```. Since the embedding encoding step consumes quite a lot of memory, we also provide the encoded file in the ```autoner_train.sh```.
- **[Optional]** Development & Test Sets.
  - Example: ```data/BC5CDR/truth_dev.ck``` and ```data/BC5CDR/truth_test.ck```
    - Three columns (i.e., token, ```Tie or Break``` label, entity type).
    - ```I``` is ```Break```.
    - ```O``` is ```Tie```.
    - Two special tokens ```<s>``` and ```<eof>``` mean the start and end of the sentence.

### Dependencies

This project is based on ```python>=3.6```. The dependent package for this project is listed as below:
```
numpy==1.13.1
tqdm
torch-scope>=0.5.0
pytorch==0.4.1
```

### Command

To train an AutoNER model, please run
```
./autoner_train.sh
```

To apply the trained AutoNER model, please run
```
./autoner_test.sh
```

You can specify the parameters in the bash files. The variables names are self-explained.


## Citation

Please cite the following two papers if you are using our tool. Thanks!

- Jingbo Shang*, Liyuan Liu*, Xiaotao Gu, Xiang Ren, Teng Ren and Jiawei Han, "**[Learning Named Entity Tagger using Domain-Specific Dictionary](https://arxiv.org/abs/1809.03599)**", in Proc. of 2018 Conf. on Empirical Methods in Natural Language Processing (EMNLP'18), Brussels, Belgium, Oct. 2018. (* Equal Contribution)
- Jingbo Shang, Jialu Liu, Meng Jiang, Xiang Ren, Clare R Voss, Jiawei Han, "**[Automated Phrase Mining from Massive Text Corpora](https://arxiv.org/abs/1702.04457)**", accepted by IEEE Transactions on Knowledge and Data Engineering, Feb. 2018.

```
@inproceedings{shang2018learning,
  title = {Learning Named Entity Tagger using Domain-Specific Dictionary}, 
  author = {Shang, Jingbo and Liu, Liyuan and Ren, Xiang and Gu, Xiaotao and Ren, Teng and Han, Jiawei}, 
  booktitle = {EMNLP}, 
  year = 2018, 
}

@article{shang2018automated,
  title = {Automated phrase mining from massive text corpora},
  author = {Shang, Jingbo and Liu, Jialu and Jiang, Meng and Ren, Xiang and Voss, Clare R and Han, Jiawei},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year = {2018},
  publisher = {IEEE}
}
```
