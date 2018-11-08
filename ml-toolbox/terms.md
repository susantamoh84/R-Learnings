#Goals of Supervised Learning
  -Find a model that best approximates f: ≈ f
  -can be Logistic Regression, Decision Tree, Neural Network ...
  -Discard noise as much as possible.
  -End goal: should acheive a low predictive error on unseen datasets.


BootStrapping - random sampling from the data
BootStrap Aggregation - Bagging - Random Forest
Boosting - 

Logistic regression - 
        -logit regression, confusionMatrix, sensitivity, specificity, accuracy, 
        -gini/ROC, KS statistics, lift/gain curve.
DecisionTree - 
        -Information Gain (IG) 
        -Criteria to measure the impurity of a node I(node):
          -gini index,
          -entropy
        -At each node, split the data based on:
          -feature f and split-point sp to maximize IG(node).
          -If IG(node)= 0, declare the node a leaf.
        
        - RegressionTree: I(node) = MSE(node) (Mean Squared Error)
          - y pred = sum(yi)/Nleaft
          
        single tree to be fit - operation - 1) restrict tree depth 2) pruning the tree. Branching done on divie n conquer basis to split to smaller size.
RandomForest - many decision trees fit on subsets of data ( including subset of features ).
BoostedTree
