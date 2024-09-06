# OPUS-BFactor

Protein B-factor, also known as the Debye-Waller temperature factor, measures the thermal fluctuation of an atom around its average position. It serves as a crucial indicator of protein flexibility and dynamics. However, accurately predicting the B-factor of Cα atoms remains challenging. In this work, we introduce OPUS-BFactor, a tool for predicting the normalized protein B-factor. The method operates in two modes: the first mode, OPUS-BFactor-seq, uses sequence information as input, allowing predictions based solely on the protein sequence; the second mode, OPUS-BFactor-struct, uses structural information, allowing predictions based on the 3D structure of target protein. Evaluation on three test sets, including recently released targets from CAMEO and CASP15, demonstrates that OPUS-BFactor significantly outperforms other B-factor prediction methods. Therefore, OPUS-BFactor is a valuable tool for predicting protein properties related to the B-factor, such as flexibility, thermal stability, and regional activity. 

We hope that OPUS-BFactor will serve as a fair baseline method in protein B-factor prediction. Additionally, the formatted datasets may become a useful benchmark to facilitate the development of protein language models, given that the performance of sequence-based B-factor prediction models still lags behind that of structure-based models.

## Usage

### Dependency

```
Python 3.7
TensorFlow 2.4
ESM-2
```

The standalone version of OPUS-BFactor is hosted on [Google Drive](xxx). The formatted datasets are hosted on [Google Drive](xxx).

## Reference 
```bibtex
@article{xu2024opus2,
  title={OPUS-BFactor: Predicting protein B-factor with sequence and structure information},
  author={Xu, Gang and Yang, Yulu and Lv, Ying and Luo, Zhenwei and Wang, Qinghua and Ma, Jianpeng},
  journal={bioRxiv},
  year={2024},
}
