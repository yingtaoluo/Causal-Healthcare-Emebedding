# Causal-Healthcare-Emebedding
Official code for "Deep Stable Representation Learning on Electronic Health Records" published and orally presented in IEEE International Conference on Data Mining (ICDM 2022). [[paper](https://doi.org/10.1109/ICDM54844.2022.00134)]  [[arXiv](https://arxiv.org/abs/2209.01321)]  

Check the [presentation material](https://github.com/yingtaoluo/Causal-Healthcare-Emebedding/blob/main/Presentation%20on%20EHR.pptx).

![image](https://github.com/yingtaoluo/Causal-Healthcare-Emebedding/blob/main/motivation.png)  
![image](https://github.com/yingtaoluo/Causal-Healthcare-Emebedding/blob/main/method.png)  

## Datasets
MIMIC III: https://physionet.org/content/mimiciii/1.4/  
MIMIC IV: https://physionet.org/content/mimiciv/0.4/  
Diagnoses and Procedures.

## Getting Started
Run [preprocessing.py](https://github.com/yingtaoluo/Causal-Healthcare-Emebedding/blob/main/preprocessing.py) for the main experiment and [preprocess_insurance.py](https://github.com/yingtaoluo/Causal-Healthcare-Emebedding/blob/main/preprocess_insurance.py) for the "OOD" experiment. Run under the [Stable](https://github.com/yingtaoluo/Causal-Healthcare-Emebedding/tree/main/Stable) folder for CHE+BaseModels and [Normal](https://github.com/yingtaoluo/Causal-Healthcare-Emebedding/tree/main/Normal) folder for BaseModels.

## Acknowledgement
Some baselines are implemented following the [PyHealth](https://github.com/sunlabuiuc/PyHealth) library.

## Citation 
If your paper benefits from this repo, please consider citing with:

```
Y. Luo, Z. Liu and Q. Liu, "Deep Stable Representation Learning on Electronic Health Records," 2022 IEEE International Conference on Data Mining (ICDM), Orlando, FL, USA, 2022, pp. 1077-1082, doi: 10.1109/ICDM54844.2022.00134.  
```

```bibtex
@INPROCEEDINGS{10027709,  
  author={Luo, Yingtao and Liu, Zhaocheng and Liu, Qiang},  
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)},   
  title={Deep Stable Representation Learning on Electronic Health Records},  
  year={2022},  
  pages={1077-1082},  
  doi={10.1109/ICDM54844.2022.00134}}

