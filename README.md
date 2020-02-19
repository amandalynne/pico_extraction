# PICO Extraction

Code for extracting PICO elements from randomized control trial abstracts. 
Built with AllenNLP and PyTorch.

`master_script.py` launches the training and evaluation pipeline, but it will not actually run unless you have an account on [Beaker](beaker.org). 

Repository structure:

pico_extraction/  
|- pico_extraction/  
|---- dataset_readers/  
|---- models/  
|---- predictors/  
|- allennlp_config/  
|- scripts/  
|- tests/  


Note! This repository does not contain the data required to run an experiment. Download it at the following links:  
[EBM_NLP](https://github.com/bepnye/EBM-NLP)  
[PICO PubMed](https://github.com/jind11/PubMed-PICO-Detection)
