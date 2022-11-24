# NLPTeamProject
NLPTeamProject

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
4. Test Using sst2.ipynb
  - Try to make the code for the evaluation, but it still doesn't work.
  - Need to think about the input type of the data.
5. Test.ipynb
  - Based on the https://github.com/timoschick/bertram -> examples
  - Make the test code, but I think it looks like predicting word.
