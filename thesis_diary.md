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

# Effect of formula on model
1.
A = torch.softmax(torch.relu(torch.mm(self.node_embeddings, self.node_embeddings.T)), dim=1)
A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])

{'accuracy': [0.25577445652173914, 0.0], 'precision_macro': [0.06394361413043478, 0.0], 'precision_weighted': [0.06542057260899102, 0.0], 'recall_macro': [0.25, 0.0], 'recall_weighted': [0.25577445652173914, 0.0], 'auroc': [0.5, 0.0], 'n_params': 358404.0}

2.
A = torch.softmax(torch.mm(self.node_embeddings, self.node_embeddings.T), dim=1)
A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])

{'accuracy': [0.25577445652173914, 0.0], 'precision_macro': [0.06394361413043478, 0.0], 'precision_weighted': [0.06542057260899102, 0.0], 'recall_macro': [0.25, 0.0], 'recall_weighted': [0.25577445652173914, 0.0], 'auroc': [0.5, 0.0], 'n_params': 358404.0}

3.
A = torch.softmax(torch.relu(torch.mm(self.node_embeddings, self.node_embeddings.T)), dim=1)
# A = F.dropout(A, 0.5)
# A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])
{'accuracy': [0.25577445652173914, 0.0], 'precision_macro': [0.06394361413043478, 0.0], 'precision_weighted': [0.06542057260899102, 0.0], 'recall_macro': [0.25, 0.0], 'recall_weighted': [0.25577445652173914, 0.0], 'auroc': [0.5, 0.0], 'n_params': 358404.0}

4.
A = torch.mm(self.node_embeddings, self.node_embeddings.T)
# A = F.dropout(A, 0.5)
A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])

{'accuracy': [0.8629415760869565, 0.002547554347826053], 'precision_macro': [0.8631925246512202, 0.0024915543063956935], 'precision_weighted': [0.8632570094306171, 0.0023508164033215206], 'recall_macro': [0.8629210190911619, 0.0025968625407968293], 'recall_weighted': [0.8629415760869565, 0.002547554347826053], 'auroc': [0.9086192260419412, 0.001714180345783023], 'n_params': 358404.0}

5. 
A = torch.relu(torch.mm(self.node_embeddings, self.node_embeddings.T))
# A = F.dropout(A, 0.5)
A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])

{'accuracy': [0.5315896739130435, 0.2758152173913043], 'precision_macro': [0.4371952174956075, 0.3732516033651727], 'precision_weighted': [0.4378306766591711, 0.37241010405018005], 'recall_macro': [0.5286948349196608, 0.27869483491966085], 'recall_weighted': [0.5315896739130435, 0.2758152173913043], 'auroc': [0.6857904820388758, 0.18579048203887588], 'n_params': 358404.0}

6. 
A = torch.mm(self.node_embeddings, self.node_embeddings.T)
# A = F.dropout(A, 0.5)
# A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])

{'accuracy': [0.8597146739130435, 0.006793478260869568], 'precision_macro': [0.8598253200928478, 0.006813400420683624], 'precision_weighted': [0.8598509356878725, 0.006860863633053771], 'recall_macro': [0.8597422232907069, 0.00681701386837652], 'recall_weighted': [0.8597146739130435, 0.006793478260869568], 'auroc': [0.9064897905135753, 0.004544208998366606], 'n_params': 358404.0}

7. Imagine auto

{'accuracy': [0.8603940217391304, 0.01392663043478265], 'precision_macro': [0.8653031631442476, 0.01082569438824893], 'precision_weighted': [0.8654758799112537, 0.01050739246207566], 'recall_macro': [0.8605162983166408, 0.013740539527024198], 'recall_weighted': [0.8603940217391304, 0.01392663043478265], 'auroc': [0.9070006593133121, 0.009169711073037878], 'n_params': 358404.0}



# Choosing subjects randomly
When picking 5 subjects for training and 1 for testing, some subjects overlap.
Originally decided to create mutually exclusive train and test sets but decided to not.

# Einsum and matmul difference\
```python
def without_einsum(input, a):
    output = []
    for i in input:
        output.append(torch.mm(a, i))
    return torch.stack(output)

def with_einsum(input, a):
    return torch.einsum("ij,kjl->kil", a, input)
```
Einsum will work but due to floating point precision some values will be different.
```python
A_temp = torch.from_numpy(A_temp)
output1 = without_einsum(input, A_temp).numpy()
output2 = with_einsum(input, A_temp).numpy()

print(np.unique(output1.round(3) == output2.round(3)))
```
returns:
```
[ True]
```

# 2021/12/15
TODO: Implemet GCRAM and GCRAMAuto
