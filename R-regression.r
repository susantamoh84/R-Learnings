# unemployment is loaded in the workspace
summary(unemployment)

# Define a formula to express female_unemployment as a function of male_unemployment
fmla <- as.formula("female_unemployment ~ male_unemployment")

# Print it
print(fmla)

# Use the formula to fit a model: unemployment_model
unemployment_model <- lm(fmla, unemployment)

# Print it
print(unemployment_model)

# broom and sigr are already loaded in your workspace
# Print unemployment_model
print(unemployment_model)

# Call summary() on unemployment_model to get more details
summary(unemployment_model)

# Call glance() on unemployment_model to see the details in a tidier form
glance(unemployment_model)

# Call wrapFTest() on unemployment_model to see the most relevant details
wrapFTest(unemployment_model)

#summary(), broom::glance(), and sigr::wrapFTest().

# unemployment is in your workspace
summary(unemployment)

# newrates is in your workspace
newrates

# Predict female unemployment in the unemployment data set
unemployment$prediction <-  predict(unemployment_model, unemployment)

# load the ggplot2 package
library(ggplot2)

# Make a plot to compare predictions to actual (prediction on x axis). 
ggplot(unemployment, aes(x = prediction, y = female_unemployment)) + 
  geom_point() +
  geom_abline(color = "blue")

# Predict female unemployment rate when male unemployment is 5%
pred <- predict(unemployment_model, newrates)
# Print it
pred

# bloodpressure is in the workspace
summary(bloodpressure)

# Create the formula and print it
fmla <- as.formula("blood_pressure~age+weight")
print(fmla)

# Fit the model: bloodpressure_model
bloodpressure_model <- lm(fmla, bloodpressure)

# Print bloodpressure_model and call summary() 
print(bloodpressure_model)
summary(bloodpressure_model)

# bloodpressure is in your workspace
summary(bloodpressure)

# bloodpressure_model is in your workspace
bloodpressure_model

# predict blood pressure using bloodpressure_model :prediction
bloodpressure$prediction <- predict(bloodpressure_model, bloodpressure)

# plot the results
ggplot(bloodpressure, aes(x=prediction, y=blood_pressure)) + 
    geom_point() +
    geom_abline(color = "blue")
    
