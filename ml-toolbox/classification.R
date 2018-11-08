# Shuffle row indices: rows
rows <- sample(nrow(Sonar))

# Randomly order data: Sonar
Sonar <- Sonar[rows,]

# Identify row to split on: split
split <- round(nrow(Sonar) * 0.6)

# Create train
train <- Sonar[1:split,]

# Create test
test <- Sonar[(split+1):nrow(Sonar),]

# Fit glm model: model
model <- glm(Class ~., train, family="binomial")

# Predict on test: p
p <- predict(model, test, type="response")

# If p exceeds threshold of 0.5, M else R: m_or_r
m_or_r <- ifelse(p>0.5, "M", "R")

# Convert to factor: p_class
p_class <- as.factor(m_or_r)

# Create confusion matrix
confusionMatrix(p_class, test$Class)

# true positive rate -> sensitivity
# true negative rate >- specificity

# If p exceeds threshold of 0.9, M else R: m_or_r
m_or_r <- ifelse(p > 0.9, "M", "R")

# Convert to factor: p_class
p_class <- as.factor(m_or_r)

# Create confusion matrix
confusionMatrix(p_class, test$Class)

######### ROC Curve ###########
library(caTools)

# Predict on test: p
p <- predict(model, test, type="response")

# Make ROC curve
colAUC(p, test$Class, plotROC=TRUE)

######### Using Caret package ##########
# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

# Train glm with custom trainControl: model
model <- train(Class ~., method="glm", Sonar, trControl=myControl)

# Print model to console
print(model)

