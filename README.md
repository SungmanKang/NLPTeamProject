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

## Procedure for pretraining and implementing DeBERTRAMa
1. preprocess data from your selected corpus
- See preprocessing data section below
2. pretrain a BERTRAM instance for your specific architecture (e.g., DeBERTa)
- See pretraining BERTRAM instance section below
- Ensure that a word embedding file is downloaded and placed in fcm/wordEmbeddings
3. Inject words with context into your desired architecture (e.g., DeBERTa) using the pretrained (fused) BERTRAM instance
- This is outlined in, and can be run from, the pretrain_BERTRAM_[model] Jupyter notebooks. These notebooks detail the code used to:
> > * Create a dictionary of words with contexts from the training buckets
> > * Inject the dictionary into the model
> > * fine-tune the model on SST-2 tasks, and evaluate their performance




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


PRETRAINING BERTRAM INSTANCES:
Information: for each instance of BERTRAM that deals with a separate model architecture (in our case BERT, RoBERTa, and DeBERTa) a separate BERTRAM instance needs to be pre-trained. This pre-training involves three separate subprocessing of pre-training each, one for the _form only_ model, one for the _context only_ model, and one for the _fused_ model. 

Each of these processes is shown in a jupyter notebook, however this can also be perforemed via command line. Note: pre-training these instances can be time-intensive, in part due to the uploading of the word embedding file used. In this case, GLoVE.6B.300d is used, but the user can download and use any pre-trained word embedding they see fit.

## KEY
* "ITEM 1": the name of the architecture the BERTRAM instance is being trained for. Originally, this contained the options 'bert' and 'roberta'. Our project has modified the BERTRAM source code to include 'deberta' as an option. NOTE: needs to be lowercase and in quotes ''
* "ITEM 2": the pre-trained model path that the BERTRAM instance is being trained for. Specifically, this is a HuggingFace path, and the options are 'bert-base-uncased', 'roberta-base', and 'microsoft/deberta-base'
* "ITEM 3": the name of the word embedding file used to pre-train BERTRAM.

## Training  form-only BERTRAM instances:
Python3 train.py --model_cls '[ITEM 1]' --bert_model '[ITEM2]' --output_dir ./outputs/FORM_DIR --train_dir ./training/ --vocab ./training/train.vwc100 --emb_file ./fcm/wordEmbeddings/[ITEM3] --num_train_epochs 20 --emb_dim 768 --train_batch_size 64 --smin 1 --smax 1 --max_seq_length 96 --mode 'form' --learning_rate 0.01 --dropout 0.1

## Training context-only BERTRAM instances:
Python3 train.py --model_cls '[ITEM 1]' --bert_model '[ITEM 2]' --output_dir ./outputs/CONTEXT_DIR --train_dir ./training/ --vocab ./training/train.vwc100 --emb_file ./fcm/wordEmbeddings/[ITEM 3] --num_train_epochs 3 --emb_dim 768 --max_seq_length 96 --mode 'context' --train_batch_size 24 --no_finetuning --smin 4 --smax 32

## Fusing BERTRAM form and context instances:
Python3 fuse_models.py --form_model ./outputs/FORM_DIR --context_model ./outputs/CONTEXT_DIR --mode 'add' --output ./outputs/FUSED_DIR


## Training Fused BERTRAM instances:
%run train.py \
    --model_cls '[ITEM 1]' \
    --bert_model ./outputs/FUSED_FIR \
    --output_dir ./outputs/FUSED_TRAINED_DIR \
    --train_dir ./training/ \
    --vocab ./training/train.vwc100 \
    --emb_file ./fcm/wordEmbeddings/[ITEM 3] \
    --emb_dim 768 \
    --mode 'add' \
    --train_batch_size 24 \
    --max_seq_length 96 \
    --num_train_epochs 3 \
    --smin 4 \
    --smax 32 \
    --optimize_only_combinator \
    --learning_rate 0.01 \
    --dropout 0.1 
