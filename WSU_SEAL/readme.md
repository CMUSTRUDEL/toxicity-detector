# Toxicity Detection (WSU-SEAL implementation)

This is a modified version of the original STRUDEL toxicity detector created by Raman et al.
In this version, we have implemented a pipeline to compute attributes for 
a new training dataset and cross-validate the model using that. Please check the WSU-SEAL 
folder for the two files for dataset preparation and cross-validation.

Data preparation: https://github.com/WSU-SEAL/toxicity-detector/blob/master/WSU_SEAL/prepare_data_strudel.py
Cross-validation: https://github.com/WSU-SEAL/toxicity-detector/blob/master/WSU_SEAL/STRUDEL_CV.py

* Please create a new anacoda environment. I used python 3.8.
* To run it please make sure that you have installed correct version of the libraries as listed 
 in the WSU-SEAL/requirements.txt. 
* The politeness model is trained using this configuration (numpy==1.23.0, scikit-learn==1.1.1, scipy==1.8.1, pandas==1.4.3)a earlier version of scikit-learn, therefore it will throw errors with a different version.
* Also, please make sure to include your perspective api key in: 
* https://github.com/WSU-SEAL/toxicity-detector/blob/master/WSU_SEAL/PPAClient.py

