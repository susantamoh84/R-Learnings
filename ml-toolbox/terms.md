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
  - K fold CV error = mean(all fold errors)
  - If CV error > training set error - f suffers from high variance
    - f is said to overfit the training set. To remedy overfitting:
      - decrease model complexity,
        - for ex: decrease max depth, increase min samples per leaf, ...
      - gather more data

  - if CV error of ≈ training set error of >> desired error - high bias problem
    - f is said to underfit the training set. To remedy underfitting:
      - increase model complexity
        - for ex: increase max depth, decrease min samples per leaf, ...
      - gather more relevant features

# DecisionTree - 
  - Information Gain (IG) 
  - Criteria to measure the impurity of a node I(node):
    - gini index,
    - entropy
  - At each node, split the data based on:
    - feature f and split-point sp to maximize IG(node).
    - If IG(node)= 0, declare the node a leaf.

  - RegressionTree: I(node) = MSE(node) (Mean Squared Error)
    - y pred = sum(yi)/Nleaft

  - single tree to be fit - operation - 1) restrict tree depth 2) pruning the tree. Branching done on divie n conquer basis to split to smaller size.

# Ensemble Learning

  - Advantages of CARTs
    - Simple to understand.
    - Simple to interpret.
    - Easy to use.
    - Flexibility: ability to describe non-linear dependencies.
    - Preprocessing: no need to standarize or normalize features, ...
    
  - Limitations of CARTs
    - Classification: can only produce orthogonal decision boundaries.
    - Sensitive to small variations in the training set.
    - High variance: unconstrained CARTs may overfit the training set.
    - Solution: ensemble learning.

  - Ensemble Learning
    - Train different models on the same dataset.
    - Let each model make its predictions.
    - Meta-model: aggregates predictions of individual models.
    - Final prediction: more robust and less prone to errors.
    - Best results: models are skillful in different ways

# Ensemble Learning in Practice: Voting Classifier
  - Binary classification task.
    - N classifiers make predictions: P , P , ..., P with P = 0 or 1.
    - Meta-model prediction: hard voting.
    
# Bagging
  - Voting Classifier
    - same training set,
    - ≠ algorithms.
  - Bagging
    - one algorithm,
    - ≠ subsets of the training set.
    
  - Bagging: Bootstrap Aggregation.
  - Uses a technique known as the bootsrap.
  - Reduces variance of individual models in the ensemble

  - Bagging: Classification & Regression
    - Classification:
      - Aggregates predictions by majority voting.
      - BaggingClassifier in scikit-learn.
    - Regression:
      - Aggregates predictions through averaging.
      - BaggingRegressor in scikit-learn
      
  - some instances may be sampled several times for one model,
  - other instances may not be sampled at all.
  
  - Out Of Bag (OOB) instances
    - On average, for each model, 63% of the training instances are sampled.
    - The remaining 37% constitute the OOB instances
    - OOB error - mean(all OOB error for individual trees)

# Random Forest

  - Bagging
    - Base estimator: Decision Tree, Logistic Regression, Neural Net, ...
    - Each estimator is trained on a distinct bootstrap sample of the training set
    - Estimators use all features for training and prediction

  - Further Diversity with Random Forests
    - Base estimator: Decision Tree
    - Each estimator is trained on a different bootstrap sample having the same size as the training set
    - RF introduces further randomization in the training of individual trees
      - d features are sampled at each node without replacement ( d < total number of features )

  - Random Forests: Classification & Regression
    - Classification:
      - Aggregates predictions by majority voting
      - RandomForestClassifier in scikit-learn
    - Regression:
      - Aggregates predictions through averaging
      - RandomForestRegressor in scikit-learn

  - Feature Importance
    - Tree-based methods: enable measuring the importance of each feature in prediction.
    - In sklearn:
      - how much the tree nodes use a particular feature (weighted average) to reduce impurity
      - accessed using the attribute feature_importance_

# Boosting

  - Boosting: Ensemble method combining several weak learners to form a strong learner.
    - Weak learner: Model doing slightly better than random guessing.
    - Example of weak learner: Decision stump (CART whose maximum depth is 1).
    
  - Train an ensemble of predictors sequentially.
  - Each predictor tries to correct its predecessor.
  - Most popular boosting methods:
    - AdaBoost,
    - Gradient Boosting.

  - Adaboost
    - Stands for Adaptive Boosting.
    - Each predictor pays more attention to the instances wrongly predicted by its predecessor.
    - Achieved by changing the weights of training instances.
    - Each predictor is assigned a coefficient α. α depends on the predictor's training error.

  - AdaBoost: Prediction
    - Classification:
      - Weighted majority voting.
      - In sklearn: AdaBoostClassifier.
    - Regression:
      - Weighted average.
      - In sklearn: AdaBoostRegressor

  - Gradient Boosted Trees
    - Sequential correction of predecessor's errors.
    - Does not tweak the weights of training instances.
    - Fit each predictor is trained using its predecessor's residual errors as labels.
    - Gradient Boosted Trees: a CART is used as a base learner.

  - Gradient Boosted Trees: Prediction
    - Regression: y = y + ηr + ... + ηr
      - In sklearn: GradientBoostingRegressor.
    - Classification:
      - In sklearn: GradientBoostingClassifier

# Stochastic Gradient Boosting (SGB)  
  - Gradient Boosting: Cons
    - GB involves an exaustive search procedure.
    - Each CART is trained to find the best split points and features.
    - May lead to CARTs using the same split points and maybe the same features.

  - Stochastic Gradient Boosting
    - Each tree is trained on a random subset of rows of the training data.
    - The sampled instances (40%-80% of the training set) are sampled without replacement.
    - Features are sampled (without replacement) when choosing split points.
    - Result: further ensemble diversity.
    - Effect: adding further variance to the ensemble of trees.

# Tuning HyperParameters

  - Machine learning model:
    - parameters: learned from data
      - CART example: split-point of a node, split-feature of a node, ...
    - hyperparameters: not learned from data, set prior to training
      - CART example: max_depth, min_samples_leaf, splitting criterion ...

  - What is hyperparameter tuning?
    - Problem: search for a set of optimal hyperparameters for a learning algorithm.
    - Solution: find a set of optimal hyperparameters that results in an optimal model.
    - Optimal model: yields an optimal score.
      - Score: in sklearn defaults to accuracy (classification) and R-square (regression).
      - Cross validation is used to estimate the generalization performance.

  - Approaches to hyperparameter tuning
    - Grid Search
    - Random Search
    - Bayesian Optimization
    - Genetic Algorithms

  - Grid search cross validation
    - Manually set a grid of discrete hyperparameter values.
    - Set a metric for scoring model performance.
    - Search exhaustively through the grid.
    - For each set of hyperparameters, evaluate each model's CV score.
    - The optimal hyperparameters are those of the model achieving the best CV score.

  - Grid search cross validation: example
    - Hyperparameters grids:
      - max_depth = {2,3,4},
      - min_samples_leaf = {0.05, 0.1}
    - hyperparameter space = { (2,0.05) , (2,0.1) , (3,0.05), ... }
    - CV scores = { score , ... }
    - optimal hyperparameters = set of hyperparameters corresponding to the best CV score.


BootStrapping - random sampling from the data
BootStrap Aggregation - Bagging - Random Forest
Boosting - 

Logistic regression - 
        -logit regression, confusionMatrix, sensitivity, specificity, accuracy, 
        -gini/ROC, KS statistics, lift/gain curve.
RandomForest - many decision trees fit on subsets of data ( including subset of features ).
BoostedTree
