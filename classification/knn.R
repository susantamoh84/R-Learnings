
# Load the 'class' package
library(class)

# Create a vector of labels
sign_types <- signs$sign_type

# Classify the next sign observed
knn(train = signs[-1], test = next_sign, cl = sign_types)

# Examine the structure of the signs dataset
str(signs)

# Count the number of signs of each type
table(signs$sign_type)

# Check r10's average red level by sign type
aggregate(r10 ~ sign_type, data = signs, mean)

# Use kNN to identify the test road signs
sign_types <- signs$sign_type
signs_pred <- knn(train = signs[-1], test = test_signs[-1], cl = sign_types)

# Create a confusion matrix of the actual versus predicted values
signs_actual <- test_signs$sign_type
table(signs_actual, signs_pred)

# Compute the accuracy
mean(signs_actual == signs_pred)

# Compute the accuracy of the baseline model (default k = 1)
k_1 <- knn(train = signs[,-1], test = signs_test[,-1], cl = sign_types)
mean(k_1 == signs_test$sign_type)

# Modify the above to set k = 7
k_7 <- knn(train = signs[,-1], test = signs_test[,-1], cl = sign_types, k = 7)
mean(k_7 == signs_test$sign_type)

# Set k = 15 and compare to the above
k_15 <- knn(train = signs[,-1], test = signs_test[,-1], cl = sign_types, k = 15)
mean(k_15 == signs_test$sign_type)

# Use the prob parameter to get the proportion of votes for the winning class
sign_pred <- knn(train = signs[,-1], test = signs_test[,-1], cl = sign_types, k = 7, prob=TRUE)

# Get the "prob" attribute from the predicted classes
sign_prob <- attr(sign_pred, "prob")

# Examine the first several predictions
head(sign_pred)

# Examine the proportion of votes for the winning class
head(sign_prob)

#normalization is performed to have equal influence of variable on the model.
