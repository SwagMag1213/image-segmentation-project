# Load necessary library
library(readr)
library(dplyr)
library(ggplot2)

setwd("C:/Users/mjohj/OneDrive - Danmarks Tekniske Universitet/Xcelligence_cells")

# Read the CSV file into a dataframe
area_results_df <- read_csv("mask_intensities_corrected.csv")

# Crop data to only consider hour > 10
#area_results_df <- area_results_df %>%
#  filter(hour > 5)

# Save data from each column into individual variables
#image_filename <- factor(area_results_df$image_filename)
well_ID <- factor(area_results_df$well_ID)
treatment <- factor(area_results_df$treatment) 
plate_position <- factor(area_results_df$plate_position)
hour <- factor(area_results_df$hour)
#mean_area <- area_results_df$mean_area 
#iou <- area_results_df$iou
mask_intensity <- area_results_df$mean_intensity_mask_adjusted

# Create the interaction plot using the natural log of mean_area
with(area_results_df, 
     interaction.plot(x.factor = hour,                        # Time (Hour)
                      trace.factor = well_ID,                 # Treatment
                      response = log(mean_area),              # Natural log of Mean Area
                      legend = FALSE,                         # No legend (set TRUE to add it)
                      las = 1,                                # Rotate y-axis labels horizontally
                      lty = rep(1:3, each = 10),              # Line types
                      col = rep(2:4, each = 10),              # Colors for treatments
                      xlab = "Time (Hour)",                   # X-axis label
                      ylab = "Natural Log of Mean Area",      # Y-axis label
                      trace.label = "Treatment"))             # Label for trace factor

# Create a numeric version of hour for continuous plotting
area_results_df$hour_num <- as.numeric(area_results_df$hour)

# Plot using ggplot2 with natural log of mean_area
ggplot(area_results_df, aes(x = hour_num, y = mean_area, group = well_ID, color = plate_position)) +
  geom_line() +
  labs(x = "Time (Hour)", y = "Mean Area", color = "Plate Position") +
  theme_minimal()

# Compute mean of mean_area 
mns <- area_results_df %>%
  group_by(treatment, hour, hour_num) %>%
  summarize(mean_area = mean(mean_area), .groups = 'drop')

# Plot the mean of mean_area (without log transformation)
ggplot(mns, aes(x = hour_num, y = mean_area, group = treatment, colour = treatment)) +
  geom_point() + 
  geom_line() +
  labs(x = "Hour", y = "Mean of Mean Area", colour = "Treatment") +
  theme_minimal()

# Plot using ggplot2 with natural log of mean_area
ggplot(area_results_df, aes(x = hour_num, y = log(mean_area), group = well_ID, color = treatment)) +
  geom_line() +
  labs(x = "Time (Hour)", y = "Natural Log of Mean Area", color = "Treatment") +
  theme_minimal()

# Compute mean log of mean_area (lnc) using dplyr
mns <- area_results_df %>%
  group_by(treatment, hour, hour_num) %>%
  summarize(lnc = mean(log(mean_area)), .groups = 'drop')

# Plot with ggplot
ggplot(mns, aes(x = hour_num, y = lnc, group = treatment, colour = treatment)) +
  geom_point() + 
  geom_line() +
  labs(x = "Time (Hour)", y = "Mean Log of Mean Area", color = "Treatment") +
  theme_minimal()

# --------- Box-Cox transformation ---------

# Load the required package
library(MASS)

if (all(mean_area > 0)) {
  
  # Fit the Box-Cox transformation
  boxcox_results <- boxcox(mean_area ~ hour, lambda = seq(-2, 2, by = 0.1))
  
  # Find the optimal lambda value (the one that maximizes the log-likelihood)
  lambda_opt <- boxcox_results$x[which.max(boxcox_results$y)]
  
  # Apply the Box-Cox transformation with the optimal lambda
  if (lambda_opt == 0) {
    mean_area_transformed <- log(mean_area)
  } else {
    mean_area_transformed <- (mean_area^lambda_opt - 1) / lambda_opt
  }
  
  # Print the optimal lambda
  cat("Optimal lambda:", lambda_opt, "\n")
  
  # Optionally: Create a transformed variable in the data frame
  area_results_df$mean_area_transformed <- mean_area_transformed
  
} else {
  cat("Error: mean_area contains zero or negative values. Box-Cox transformation requires positive values.")
}

# Visual check with histogram and Q-Q plot
par(mfrow = c(1, 2))  # Set up for two plots side by side

# Histogram of transformed data
hist(area_results_df$mean_area_transformed, main = "Histogram of Box-Cox Transformed Data", 
     xlab = "Transformed Mean Area", col = "lightblue", border = "black")

# Q-Q plot of transformed data
qqnorm(area_results_df$mean_area_transformed, main = "Q-Q Plot of Box-Cox Transformed Data")
qqline(area_results_df$mean_area_transformed, col = "red")

# Reset plotting area
par(mfrow = c(1, 1))

# Statistical test for normality (Shapiro-Wilk test)
shapiro_test <- shapiro.test(area_results_df$mean_area_transformed)
print(shapiro_test)

# Calculate the mean of transformed mean_area for each hour
mean_by_hour <- aggregate(mean_area_transformed ~ hour, data = area_results_df, FUN = mean)

# Scatter plot for mean_area_transformed
ggplot(area_results_df, aes(x = hour, y = mean_area_transformed)) +
  geom_point(color = "blue", alpha = 0.5) +  # Scatter points
  geom_line(data = mean_by_hour, aes(x = hour, y = mean_area_transformed), color = "red", linewidth = 1.2) +  # Line for mean by hour
  labs(title = "Transformed Mean Area vs Hour with Mean by Hour", 
       x = "Hour", 
       y = "Transformed Mean Area") +
  theme_minimal()

# --------- Separate analysis for each time-point ---------
library(lme4)
library(lmerTest)
# Function to fit model, plot residuals, and return ANOVA results
fn <- function(df) {
  # Fit the model with natural log of mean_area
  model <- lmer(mask_intensity ~ (1|plate_position) + treatment, data = df)
  #model <- lm(mask_intensity ~ plate_position, data = df)
  
  # Get the ANOVA table and extract F-value and p-value
  anova_result <- anova(model)
  
  # Extract residuals and fitted values from the model
  residuals <- residuals(model)
  fitted_values <- fitted(model)
  
  # Set up a 1x3 plotting area to include histogram, Q-Q plot, and residuals vs. fitted plot
  par(mfrow = c(1, 3))  # Set up the plotting area: 1 row, 3 columns
  
  # Histogram of residuals
  hist(residuals, main = paste("Residuals for Hour", unique(df$hour)), 
       xlab = "Residuals", col = "lightblue", border = "black")
  
  # Q-Q plot for normality of residuals
  qqnorm(residuals, main = paste("Q-Q Plot for Hour", unique(df$hour)))
  qqline(residuals, col = "red")
  
  # Residuals vs. Fitted Values plot
  plot(fitted_values, residuals, 
       main = paste("Residuals vs. Fitted for Hour", unique(df$hour)),
       xlab = "Fitted Values", ylab = "Residuals",
       pch = 19, col = "blue")
  abline(h = 0, col = "red", lwd = 2)  # Add a horizontal line at y=0
  
  # Return the F-value and p-value from the ANOVA
  return(unlist(anova_result[1, c("F value", "Pr(>F)")]))
}

# Split the data frame by 'hour' into a list of data frames
hour_data <- split(area_results_df, f = area_results_df$hour)

# Apply the function to each data frame in the list and round results
anova_results <- round(sapply(hour_data, fn), 3)

for(i in 1:length(hour_data)){
  fn(hour_data[[i]])
  print(i)
}

# View the results
print(anova_results)


# --------- Random effects approach I ---------
library(lme4)

# Fit the mixed-effects model
model1 <- lmer(mean_area ~ hour + treatment + hour:treatment + (1 | well_ID), data = area_results_df)

# Extract residuals from the model
residuals_model1 <- resid(model1)

# Plot histogram of residuals
ggplot(data.frame(residuals = residuals_model1), aes(x = residuals)) +
  geom_histogram(binwidth = 0.2, fill = "lightblue", color = "black") +
  labs(title = "Histogram of Residuals", x = "Residuals", y = "Frequency") +
  theme_minimal()

# QQ plot of residuals
ggplot(data.frame(sample = residuals_model1), aes(sample = sample)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

fitted_values <- fitted(model1)
residuals <- resid(model1)

# Create the residuals vs. fitted plot
ggplot(area_results_df, aes(x = fitted_values, y = residuals)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs. Fitted Values") +
  theme_minimal()

# ANOVA with Satterthwaite approximation for degrees of freedom
anova(model1)

VarCorr(model1)

c(-2 * logLik(model1, REML=TRUE)) # REML=TRUE is default


# --------- Random effects approach I, same but different ---------
library(nlme)
library(car)

model1 <- gls(mean_area ~ hour * treatment, correlation=corCompSymm(form= ~ 1|well_ID), data=area_results_df)

# Extract residuals from the model
residuals_model1 <- resid(model1)

# Plot histogram of residuals
ggplot(data.frame(residuals = residuals_model1), aes(x = residuals)) +
  geom_histogram(binwidth = 0.2, fill = "lightblue", color = "black") +
  labs(title = "Histogram of Residuals", x = "Residuals", y = "Frequency") +
  theme_minimal()

# QQ plot of residuals
ggplot(data.frame(sample = residuals_model1), aes(sample = sample)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

fitted_values <- fitted(model1)
residuals <- resid(model1)

# Create the residuals vs. fitted plot
ggplot(area_results_df, aes(x = fitted_values, y = residuals)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs. Fitted Values") +
  theme_minimal()

summary(model1)

anova(model1)

car::qqPlot(residuals)


# --------- GAUSSIAN MODEL OF SPATIAL CORRELATION ---------
# library(glmer)
model2 <- lme(mean_area_transformed ~ hour*treatment + I(hour^2),
              random= ~ 1|well_ID,
              correlation=corARMA(form= ~ hour_num|well_ID, p = 1, q = 1),#, nugget=TRUE),
              data=area_results_df)

# model2 <- lme(mean_area_transformed ~ hour*treatment,
#               random= ~ 1|well_ID,
#               correlation=corGaus(form= ~ hour_num|well_ID, nugget=TRUE),
#               data=area_results_df)

# Extract residuals from the model
residuals_model2 <- resid(model2)

# Plot histogram of residuals
ggplot(data.frame(residuals = residuals_model2), aes(x = residuals)) +
  geom_histogram(binwidth = 0.2, fill = "lightblue", color = "black") +
  labs(title = "Histogram of Residuals 2", x = "Residuals", y = "Frequency") +
  theme_minimal()

# QQ plot of residuals
ggplot(data.frame(sample = residuals_model2), aes(sample = sample)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ Plot of Residuals 2", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

fitted_values <- fitted(model2)
residuals <- resid(model2)



# Create the residuals vs. fitted plot
ggplot(area_results_df, aes(x = fitted_values, y = residuals)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs. Fitted Values") +
  theme_minimal()

resid2 <- c(ranef(model2))$`(Intercept)`

car::qqPlot(residuals)

acf(residuals)
pacf(residuals)

summary(model2)

intervals(model2, which = "var-cov")

anova(model2)

anova(model1, model2)

# --------- Inspect Variograms ---------

plot(Variogram(model2, form= ~ hour_num | well_ID, data=area_results_df))

model3 <- lme(log(mean_area) ~ hour * treatment, random= ~ 1 | well_ID,
              correlation=corExp(form= ~ hour_num | well_ID, nugget = FALSE),
              data = area_results_df)

plot(Variogram(model3, form = ~as.numeric(hour) | well_ID, data = area_results_df))


