# Pre-Processing cheat sheet

  - Start with median imputation
  - For Linear models
    - Center and scale
    - Try PCA and spatial sign
  - Tree-based models don't need much pre-processing

# No ( or low ) variance variables

  - Some variables don't contain much information
    - Constant ( i.e no variance )
    - Nearly Constant ( i.e low variance )
  - Easy for one fold of CV to end up with constant column
  - Can cause problems for your models
  - Usually remove extremely low variance variables
  - caret package "zv" argument in pre-processing to solve this issue.
  

# Principle components analysis

  - Combines low-variance and correlated variables
  - Single set of high-variance, perpendicular predictors
  - Prevents collinearity ( i.e correlation among predictors )
  
# Options - 
  - zv, center, scale
  - nzv, center, scale
  - zv, center, scale, pca -> Best results.

