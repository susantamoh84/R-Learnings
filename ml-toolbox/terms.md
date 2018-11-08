# Goals of Supervised Learning
  - Find a model that best approximates f: ≈ f
  - can be Logistic Regression, Decision Tree, Neural Network ...
  - Discard noise as much as possible.
  - End goal: should acheive a low predictive error on unseen datasets.

# Difficulties in Approximating f
  - Overfitting: (x) fits the training set noise.
  - Underfitting: is not flexible enough to approximate f
  
# Generalization Error
  - Generalization Error of : Does generalize well on unseen data?
  - It can be decomposed as follows:
    - Generalization Error of = bias^2 + variance + irreducible error
  - Bias: error term that tells you, on average, how much f ≠ f.
  - Variance: tells you how much is inconsistent over different training sets

# Model Complexity
  - Model Complexity: sets the flexibility of .
  - Example: Maximum tree depth, Minimum samples per leaf
  
# Bias-Variance Tradeoff
  - The optimum model complexity
  
# Estimating the Generalization Error
  - Solution:
    - split the data to training and test sets,
    - fit to the training set,
    - evaluate the error of on the unseen test set.
    - generalization error of f ≈ test set error of f.
    
# Better Model Evaluation with Cross-Validation
  - Test set should not be touched until we are confident about's performance.
  - Evaluating on training set: biased estimate, has already seen all
      - training points.
  - Solution → Cross-Validation (CV):
      - K-Fold CV,
      - Hold-Out CV.


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
