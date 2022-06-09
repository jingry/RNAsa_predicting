

The code in this repository is for predicting RNA solvent accessibility, multiple models used for training and hundreds of checkpoints were kept for using. The basic usage is 

`python <modelFile> <TestDataFile> <CheckPointFile> <EmbeddingSize> <DropoutRate>`

For example:
`python GlobalAtt.py TestData/TS31.pk CheckPoints/checkpoints_round1/GlobalAtt_TR95_16_0.20.pt 16 0.2`

Below are the brief explanation of the parameters:
modelFile: the file contains the model, currently the .py files in the root path of the repository are available.
TestDataFile: the saved data for testing, currently three files in folder TestData are available.
CheckPointFile: the saved checkpoint file, currently the 405 files in folder CheckPoints are available. 
EmbeddingSize: integer, only 16, 32 or 64 are available, please note that this value should be the same with CheckPointFile. 


NOTE: The format of the saved checkpoint files are ‘ModelName_TestFile_ EmbeddingSize_ DropoutRate’, thus please specify the paths accordingly, otherwise the predicting might be wrong. We didn’t provide the automatic script is because we provided the origin model, thus the current usage will be better for further development.


Required extra packages:

Pytorch>=1.10.0
scikit-learn
