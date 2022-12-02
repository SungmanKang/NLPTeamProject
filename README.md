# NLPTeamProject
NLPTeamProject

PAPERS:

DeBERTa - https://arxiv.org/abs/2006.03654
BERTRAM - https://arxiv.org/abs/1910.07181


GitHub Repositories (much of the source code used in DeBERTRAMa is modified from these repositories, with changes made to account for new word embeddings, DeBERTa's architcture, and syntax changes in the modern Transformers library on HuggingFace):

DeBERTa: https://github.com/microsoft/DeBERTa
BERTRAM : https://github.com/timoschick/bertram


HuggingFace Model Repositories (the models created for this project are currently hosted on HuggingFace, and can be accessed directly through their website, or interected with via the Transfomers library):

Huggingface : https://huggingface.co/rcorkill


11/23/2022
1. Download SST-2 dataset for testing.
  - https://gluebenchmark.com/tasks/
  - The Stanford Sentiment Treebank
  - Zip this folder for uploading on the Jupyternotebook
  - Unzip using code that is almost on top of the runner.ipynb file, and use the path for testing.
2. Upload run_glue.py
  - Changed the transformers version check part : 4.25.0.dev0 -> 4.24.0
  - To run this file, installed the evaluate : !pip install evaluate
3. test run_glue
  - Even if I changed the train_file, there is no difference as before.
  - Change the epoch and try to use other parameters.
4. Test Using sst2.ipynb
  - Try to make the code for the evaluation, but it still doesn't work.
  - Need to think about the input type of the data.
5. Test.ipynb
  - Based on the https://github.com/timoschick/bertram -> examples
  - Make the test code, but I think it looks like predicting word.


PREPROCESSING DATA:
- This only needs to be done once for your corpus, and constructs training buckets to be used when pre-traning any instances of BERTRAM
run: Python3 fcm/preprocess.py train --input ./fcm/brown/brown.txt --output ./training
-input is the path of your text corpus (in our example, brown) and --ouput directs to your training directory, where the training buckets will be saved
