# Sep 1
 - First proper model made
 - Outperforms both FCN, CNN, RNN in 5 subject dataset
   - But unreliable because accuracy slightly above statistical average
 - Might need to find new dataset to experiment
 - GCN Auto model not working well in 108 subject dataset
   - Could be because adj is different for each subject



## Comments from huang
learn from compilation of graphs, cycle, random, tree, grid, path etc, types of learning based on graphs

sparse, plot the times series of predictions and do feature engineering based on
    Denoise 
    Normalize based on eeg rest state
    Fourier denoise1 (spectral)

Try graph regression on eeg dataset

1. GCN instead GCGRU, TGN
2. Regression on EEG dataset

3. Dataset with graph classification, same model 
4. Feature engineering
5. Learning from compilation gives answer to which type of graph works for EEG


GCN smoothes graph, denoise2 (eigen)

