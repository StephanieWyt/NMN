# NMN

Source code and datasets for ACL 2020 paper: [***Neighborhood Matching Network for Entity Alignment***](https://arxiv.org/pdf/2005.05607.pdf).

## Datasets

> Please first download the datasets [here](https://drive.google.com/drive/folders/1SN3JAV3clMMUPQ0M6LTJQ4GZ8JFLTy0s?usp=sharing) and extract them into `data/` directory.

Initial datasets DBP15K and DWY100K are from [JAPE](https://github.com/nju-websoft/JAPE) and [BootEA](https://github.com/nju-websoft/BootEA).

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG (DBP_ZH);
* triples_1_s: remaining relation triples encoded by ids in source KG (S-DBP_ZH);
* triples_2: relation triples encoded by ids in target KG (DBP_EN);
* triples_2_s: remaining relation triples encoded by ids in target KG (S-DBP_EN);
* vectorList.json: the input entity feature matrix initialized by word vectors;

## Environment

* Python>=3.5
* Tensorflow>=1.8.0
* Scipy
* Numpy

## Running

For example, to run NMN on S-DBP15K (ZH-EN), use the following script:
```
python3 main.py --dataset S-DBP15k --lang zh_en
```


> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to wyting@pku.edu.cn.

## Citation

If you use this model or code, please cite it as follows:

*Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang and Dongyan Zhao. Neighborhood Matching Network for Entity Alignment. In: ACL 2020.*
