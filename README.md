# BCI_KER-Challenge-2015
This is my solution to the P-300 BCI_KER challenge on kaggle (https://www.kaggle.com/competitions/inria-bci-challenge/overview)

## Problem Description
As humans think, we produce brain waves. These brain waves can be mapped to actual intentions. In this competition, you are given the brain wave data of people with the goal of spelling a word by only paying attention to visual stimuli. The goal of the competition is to detect errors during the spelling task, given the subject's brain waves. 

## The Setup
The “P300-Speller” is a well-known brain-computer interface (BCI) paradigm which uses Electroencephalography (EEG) and the so-called P300 response evoked by rare and attended stimuli in order to select items displayed on a computer screen. In this experiment, each subject was presented with letters and numbers (36 possible items displayed on a matrix) to spell words. Each item of a word is selected one at a time, by flashing screen items in group and in random order. The selected item is the one for which the online algorithm could most likely recognize the typical target response.

The goal of this challenge is to determine when the selected item is not the correct one by analyzing the brain signals after the subject received feedback.

## Approach
Since the problem is over a decade old there have been multiple solutions proposed to the solution using various Machine Learning models but none until now(atleast to my knowledge) have used deep learning using multi layer perceptron(reffered to as MLP from hereon).
In this approach I have used a MLP to discriminate between the positive and negative feedbacks to allow for automatic error correction as discussed in the original paper^(1). Previous approaches used various other models or a combination of them (SVM , Logical Regression etc) however I believe that these models are inherently at an disadvantage because there are a few features(This is an intution, the relevant fetures will be updated here once enough evidence is aquired) which they are unable to take into account which the DL model is able to benfit upon without explicitly stating them due to its architeture.

## Feature Selection
The features for the model which the MLP takes as input are listed below - 
1. 11200 EEG signals which are taken in the duration from 300ms after the feedback(the amount of time it takes for the brain to register the feedback) to 1300ms after the feedback(which is the total duration for which the feedback is present on the screen). With a sampling rate of 200 Hz and 56 EEG channels that amounts to total of 11200 EEG data points for each feedback event.
2. 200 EOG signals which correspond to the eye movement of the subject in response to the feedback. Sampling rate of 200Hz.
3. 1 Time Stamp corresponding to the time of the feedback event from the beginning.(Meta Feature)
4. 1 Session Number of the subject(Meta Feature)
5. 1 Word Number(Meta Feature)
6. 1 Character Number in the word(Meta Feature)

## MLP Model Parameters
The libraries used in the overall model were - Scikit-learn, Pandas, Numpy, os
The MLP model was prepared using the sklearn neural network library with the parameters as follows-
 hidden_layer_sizes=(11404,1000,1000,2)
  1.  activation='relu'
  2.  solver='adam'
  3.  alpha = 3e-5
  4.  batch_size=200
      learning_rate='adaptive'
  5.  learning_rate_init=0.01
  6.  max_iter=200
  7.  shuffle=True
  8.  random_state=42
  9.  tol=1e-4
  10.  verbose=True
  11.  warm_start=True
  12.  early_stopping=True
  13.  validation_fraction=0.2
  14.  beta_1=0.9
  15.  beta_2=0.999
  16.  epsilon=1e-8
  17.  n_iter_no_change=10


## Refrences
1. https://onlinelibrary.wiley.com/doi/10.1155/2012/578295
2. https://github.com/alexandrebarachant/bci-challenge-ner-2015/blob/master/README.md
