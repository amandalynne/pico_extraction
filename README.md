# pico_extraction

Code for extracting PICO elements from randomized control trial abstracts. 
Built with AllenNLP and PyTorch.

Repository structure:

pico_extraction/  
|- pico_extraction/  
|---- dataset_readers/  
|---- models/  
|---- predictors/  
|- allennlp_config/  
|- scripts/  
|- tests/  
        

[EBM_NLP](https://github.com/bepnye/EBM-NLP) comes pre-split into 4802 train, 191 test examples (4993 total) &
 in their repo, they split that 4802 train into 20% dev, rest train. So the
 final split is:  
 3842 train documents (about 77%)  
 960 dev documents (about 19%)  
 191 test documents (about 4%)
