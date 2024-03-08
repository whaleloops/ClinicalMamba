# ClinicalMamba

This repository contains the implementation of prompt-based fine-tuning CclinicalMamba on n2c2 2018 shared task 1: [Cohort Selection for Clinical Trials](https://www.semanticscholar.org/paper/Cohort-selection-for-clinical-trials%3A-n2c2-2018-1-Stubbs-Filannino/29dfdb6bf2b44ea57525a6b89b72cb74413fb5a5).

This is a classification task that identifies which patients meet and do not meet the identified selection criteria given in their longitudinal clinical notes.

## Dependencies

* python=3.9.18
* numpy=1.26.3
* transformers=4.36.2
* tokenizers=0.15.0
* mamba-ssm=1.1.2
* causal-conv1d=1.1.1
* pytorch=2.1.2
* pytorch-cuda=12.1
* scikit-learn=1.4.0 

Full environment setting is lised [here](conda-environment.yaml) and can be installed through:

```
conda env create -f conda-environment.yaml
conda activate mixtral
```

## Download / preprocess data
1. Download [raw n2c2 data folder](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) train and n2c2-t1_gold_standard_test_data, and put them under ./data
2. Proprcesss the data by running the notebook: ./preprocess.ipynb. It will transform from xml to json format, where each instance is a dictionary input is 'text' and output should start with ‘label’. Example in image below:

3. Define your labels and associated prompts here ./config_labels.py. Example in image below:

4. The model then learns to assign yes or no to each prompt:



## Train and Eval
TODO

## Citation

## License

See the [LICENSE](LICENSE) file for more details.
