# Baselines for Comparison with the LSTM Encoder-Decoder
This directory contains three baselines to compare the performance of the LSTM Encoder-Decoder against.

## Lead-N
The Lead-N baseline simply extracts the first *N* sentences from a given article to create a summary. 
To experiment with this baseline, edit the contents of the `n_values` array in `test_leadn.py` to select
the values of *N* you'd like to test. Then, running the following command will determine the best of the 
`n_values` best on validation set performance, and will evaluate the test set performance:
```sh
python test_leadn.py
```
The results will be stored in a file called `scores_leadn.csv` in this directory, and will be overwritten at each run.

## Extended SumBasic
Read our article at cs.mcgill.ca/~hwiltz/literature for more details about how this works. Essentially, SumBasic determines *relevant* sentences according to the frequency of the words in each sentence, and creates summaries with the most relevant sentences. <br>
As described in the article, the SumBasic model here is split into two slightly different models, the sentences over tokens model and the tokens over sentences model. The former is in `sumbasic.py`, and the latter is implemented in `sumbasic_tokensfirst.py`.<br>
To experiment with these models, import **one of** `sumbasic` and `sumbasic_tokensfirst` to select between the two models. Then, edit the `summary_lengths`, `n_grams`, and `lambdas` lists with your choice of hyperparameter configurations. To run model selection and evaluate test set performance, run the following command:
```sh
python test_sumbasic.py
```
The results will be stored in a file called `scores_sumbasic.csv` in this directory, and will be overwritten at each run. Furthermore, the generated summaries will be stored in `summaries.txt` in this directory.
