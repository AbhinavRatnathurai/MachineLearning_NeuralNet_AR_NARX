# Load libraries
library(readxl)
library(neuralnet)
library(dplyr)
library(ggplot2)
library(cowplot)

# Read data
data <- read_excel("uow_consumption.xlsx")
# Calculate z-score for each value in the column
z_scores <- scale(data$`0.83333333333333337`)

# Identify outliers as those with absolute z-score > 3
outlier_indices <- which(abs(z_scores) > 2)

# Remove outliers from the data
data_clean <- data[-outlier_indices, ]
print(data_clean)

# Remove missing and infinite values
data_clean <- data_clean[is.finite(data_clean$`0.83333333333333337`),]

# Plot original and cleaned data
if (all(is.finite(data_clean$`0.83333333333333337`))) {
  par(mfrow = c(2, 1))
  plot(data$`0.83333333333333337`, type = "l", main = "Original Data")
  plot(data_clean$`0.83333333333333337`, type = "l", main = "Cleaned Data")
}

# Add a small constant to output data to avoid division by zero
const <- 0.01
data$`0.83333333333333337` <- data$`0.83333333333333337` + const

# Split data into training and testing sets
train_data <- data[1:380,]
test_data <- data[381:470,]


# Function to create input/output matrices
create_io_matrix <- function(data, max_lag = 4, include_lag_7 = TRUE) {
  n <- nrow(data)
  input_columns <- c()
  
  for (i in 1:max_lag) {
    input_columns <- c(input_columns, paste0("Lag_", i))
    data[[paste0("Lag_", i)]] <- c(rep(NA, i), data$`0.83333333333333337`[1:(n - i)])
  }
  
  if (include_lag_7) {
    data$Lag_7 <- c(rep(NA, 7), data$`0.83333333333333337`[1:(n - 7)])
    input_columns <- c(input_columns, "Lag_7")
  }
  
  # Remove rows with missing values
  data <- data[complete.cases(data),]
  
  return(list(inputs = data[, input_columns], output = data$`0.83333333333333337`))
}

# Create input/output matrices for training and testing
train_io <- create_io_matrix(train_data)
test_io <- create_io_matrix(test_data)

# Define a function to normalize data
normalize <- function(data) {
  return((data - min(data)) / (max(data) - min(data)))
}

# Define a function to denormalize data
denormalize <- function(norm_data, original_data) {
  return(norm_data * (max(original_data) - min(original_data)) + min(original_data))
}

# Normalize input/output matrices
train_io_norm <- list(inputs = apply(train_io$inputs, 2, normalize),
                      output = normalize(train_io$output))
test_io_norm <- list(inputs = apply(test_io$inputs, 2, normalize),
                     output = normalize(test_io$output))



#Training
train_test_mlp <- function(train_io, test_io, hidden_layers, threshold = 0.1, stepmax = 1e+06) {
  mlp <- neuralnet(output ~ ., data = cbind(train_io$inputs, output = train_io$output),
                   hidden = hidden_layers, linear.output = TRUE, threshold = threshold, stepmax = stepmax)
  
  # Test the model
  predictions_norm <- predict(mlp, test_io$inputs)
  predictions <- denormalize(predictions_norm, test_io$output)
  
  # Calculate statistical indices
  error <- test_io$output - predictions
  rmse <- sqrt(mean(error^2))
  mae <- mean(abs(error))
  mape <- mean(abs(error / (test_io$output + 1e-8))) * 100  # Add small constant to denominator
  smape <- mean(abs(error) / (abs(test_io$output) + abs(predictions))) * 200
  
  return(list(mlp = mlp, rmse = rmse, mae = mae, mape = mape, smape = smape))
}


mlp_models <- list()

hidden_layer_configurations <- list(c(5), c(10), c(15), c(5, 5), c(10, 5), c(5, 10),
                                    c(10, 10), c(15, 15), c(5, 5, 5), c(10, 5, 5),
                                    c(5, 10, 5), c(5, 5, 10))

for (i in 1:length(hidden_layer_configurations)) {
  mlp_model <- train_test_mlp(train_io_norm, test_io_norm, hidden_layers = hidden_layer_configurations[[i]])
  mlp_models[[i]] <- mlp_model$mlp
}


# Create a comparison table
comparison_table <- data.frame(
  Model = 1:length(mlp_models),
  Description = sapply(hidden_layer_configurations, function(x) paste(x, collapse = "-")),
  RMSE = sapply(mlp_models, function(mlp) {
    predictions_norm <- predict(mlp, test_io_norm$inputs)
    predictions <- denormalize(predictions_norm, test_io$output)
    sqrt(mean((test_io$output - predictions)^2))
  }),
  MAE = sapply(mlp_models, function(mlp) {
    predictions_norm <- predict(mlp, test_io_norm$inputs)
    predictions <- denormalize(predictions_norm, test_io$output)
    mean(abs(test_io$output - predictions))
  }),
  MAPE = sapply(mlp_models, function(mlp) {
    predictions_norm <- predict(mlp, test_io_norm$inputs)
    predictions <- denormalize(predictions_norm, test_io$output)
    mean(abs((test_io$output - predictions) / (test_io$output + 1e-8))) * 100
  }),
  sMAPE = sapply(mlp_models, function(mlp) {
    predictions_norm <- predict(mlp, test_io_norm$inputs)
    predictions <- denormalize(predictions_norm, test_io$output)
    mean(abs(test_io$output - predictions) / (abs(test_io$output) + abs(predictions))) * 200
  })
)

# Add a new column for accuracy percentage
comparison_table$Accuracy <- round(100 - comparison_table$MAPE, 2)

# Print the comparison table
print(comparison_table)


# Calculate the total number of weight parameters for each model
total_weight_parameters <- sapply(mlp_models, function(mlp) {
  input_features <- dim(train_io_norm$inputs)[2]
  output_features <- 1
  weight_parameters <- 0
  
  for (i in 1:length(mlp$weights)) {
    weight_parameters <- weight_parameters + length(mlp$weights[[i]])
  }
  
  return(weight_parameters)
})

# Add the total number of weight parameters to the comparison table
comparison_table$Weight_Parameters <- total_weight_parameters

# Print the updated comparison table
print(comparison_table)

# Find the best one-hidden layer and two-hidden layer networks
best_one_hidden_layer_idx <- which.min(comparison_table$RMSE[comparison_table$Description %in% c("5", "10", "15")])
best_two_hidden_layer_idx <- which.min(comparison_table$RMSE[comparison_table$Description %in% c("5-5", "10-5", "5-10", "10-10", "15-15")])

# Get the best one-hidden layer and two-hidden layer networks
best_one_hidden_layer <- mlp_models[[best_one_hidden_layer_idx]]
best_two_hidden_layer <- mlp_models[[best_two_hidden_layer_idx]]

# Compare the total number of weight parameters
cat("Best one-hidden layer network has", comparison_table$Weight_Parameters[best_one_hidden_layer_idx], "weight parameters.\n")
cat("Best two-hidden layer network has", comparison_table$Weight_Parameters[best_two_hidden_layer_idx], "weight parameters.\n")





# Find the best model based on RMSE and MAPE
best_model_idx_rmse <- which.min(comparison_table$RMSE)
best_model_idx_mape <- which.min(comparison_table$MAPE)

# Get the best model based on RMSE
best_model_rmse <- mlp_models[[best_model_idx_rmse]]
predictions_norm_rmse <- predict(best_model_rmse, test_io_norm$inputs)

predictions_rmse <- denormalize(predictions_norm_rmse, test_io$output)

# Plot actual vs. predicted values for the best model based on RMSE
par(mfrow = c(1,1))
plot(test_data$`0.83333333333333337`, type = "l", col = "blue", main = "Actual vs. Predicted Energy Consumption", xlab = "Time", ylab = "Energy Consumption")
lines(predictions_rmse, col = "red")

# Get the best model based on MAPE
best_model_mape <- mlp_models[[best_model_idx_mape]]
predictions_norm_mape <- predict(best_model_mape, test_io_norm$inputs)

predictions_mape <- denormalize(predictions_norm_mape, test_io$output)

# Plot actual vs. predicted values for the best model based on MAPE
par(mfrow = c(1, 1))
plot(test_io$output, type = "l", main = "Actual vs. Predicted Energy Consumption (MAPE)", xlab = "Time", ylab = "Energy Consumption", col = "blue")
lines(predictions_mape, col = "red")
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, cex = 0.8)



#Save the best model to disk
saveRDS(best_model_rmse, file = "best_model_rmse.rds")
saveRDS(best_model_mape, file = "best_model_mape.rds")

# Install and load the packages
if (!requireNamespace("neuralnet", quietly = TRUE)) {
  install.packages("neuralnet")
}
library(neuralnet)

if (!requireNamespace("grid", quietly = TRUE)) {
  install.packages("grid")
}
library(grid)

if (!requireNamespace("gridExtra", quietly = TRUE)) {
  install.packages("gridExtra")
}
library(gridExtra)

# Save the neural network plot as a PNG file
for (i in 1:length(hidden_layer_configurations)) {
  mlp_model <- train_test_mlp(train_io_norm, test_io_norm, hidden_layers = hidden_layer_configurations[[i]])
  mlp_models[[i]] <- mlp_model$mlp
  
  filename <- paste0("neural_network_plot_", i, ".png")
  png(filename)
  
  # Adjust the margins around the plotting area
  par(mar = c(13, 10, 10, 7) + 0.1)  # Default values with a small increase to prevent cutoff
  
  # Plot the neural network using the neuralnet package
  plot(mlp_model$mlp, rep="best")
  
  dev.off()
}





