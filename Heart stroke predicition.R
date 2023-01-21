#Group project-Satish Tirumalapudi|Sandhya sree Agolu|Harini Prabha Baskar jayanthi|
#             Pratyusha Parimi|Salma Chanbasha Nandikotkur|Anwar Shaikh
##Heart stroke Prediction Analysis

install.packages("tidyverse")
library(tidyverse)
library(data.table)
library(caret)
library(pROC)
#install.packages("cvms")
#install.packages("imbalance")

library(imbalance)
##set directory
setwd("C:/Users/satya/OneDrive/Desktop/CIS8695")

##Data Loading
data <- read.csv("healthcare-dataset-stroke-data.csv", na.strings = c('N/A'))
data <- as.data.table(data)

str(data)

#Data Cleaning and preprocessing 
#Drop ID cols
data$id <- NULL

# Check cols with NA
colnames(data)[colSums(is.na(data)) > 0]


# Get BMI per gender
mean_bmi_per_gender <- data %>% group_by(gender) %>% summarise(bmi = mean(bmi, na.rm = TRUE))


# Replace NA in BMI with the mean for each gender
data[gender == 'Female' & is.na(data$bmi), bmi := mean_bmi_per_gender[1, 'bmi']]
data[gender == 'Male'   & is.na(data$bmi), bmi := mean_bmi_per_gender[2, 'bmi']]
data[gender == 'Other'  & is.na(data$bmi), bmi := mean_bmi_per_gender[3, 'bmi']]


#Exploratory Data Anlaysis
colors <- c("tomato", "royalblue", "olivedrab1")


tbl <- with(data, table(gender, stroke))

barplot(tbl, legend = TRUE, beside = TRUE, col = colors,
        names.arg = c("No Stroke", "Stroke"), main = "Stroke events by gender")


colors <- c("tomato", "royalblue", "olivedrab1", "mediumpurple", "turquoise")

tbl <- with(data, table(work_type, stroke))

barplot(tbl, legend = TRUE, beside = TRUE, col = colors,
        names.arg = c("No Stroke", "Stroke"), main = "Stroke events by patient's work type")

barplot(tbl[, 2], col = colors, main = "Confirmed stroke events by patient's work type")


colors <- c("tomato", "royalblue")

tbl <- with(data, table(Residence_type, stroke))

barplot(tbl, legend = TRUE, beside = TRUE, col = colors, 
        names.arg = c("No Stroke", "Stroke"),
        main = "Stroke events by patient's Residence type")


barplot(tbl[, 2], col = colors,
        main = "Confirmed stroke events by patient's Residence type")

tbl <- with(data, table(age, stroke))

barplot(tbl[, 1], col = "royalblue", main = "Patients without stroke by age")



barplot(tbl[, 2], col = "tomato", main = "Patients with stroke events by age")
colors <- c("tomato", "royalblue", "olivedrab1", "mediumpurple")

tbl <- with(data, table(smoking_status, stroke))

barplot(tbl, legend = TRUE, beside = TRUE, col = colors,
        names.arg = c("No Stroke", "Stroke"), main = "Stroke events by smoking habits")

barplot(tbl[, 2], col = colors, 
        main = "Confirmed stroke events by smoking habits")


colors <- c("royalblue", "tomato")

tbl <- with(data, table(hypertension, stroke))

barplot(tbl, legend = TRUE, legend.text = c("Hypertension", "No Hypertension"), 
        beside = TRUE, col = colors,
        names.arg = c("No Stroke", "Stroke"), 
        main = "Stroke events by hypertension diagnosis")

barplot(tbl[, 2], col = colors,
        main = "Confirmed stroke events by hypertension diagnosis",
        names.arg = c("Without Hypertension", "With Hypertension"))


colors <- c("royalblue", "tomato")

tbl <- with(data, table(heart_disease, stroke))

barplot(tbl, legend = TRUE, legend.text = c("Without heart disease", "With heart disease"),
        beside = TRUE, col = colors,
        names.arg = c('No Stroke', 'Stroke'), 
        main = "Stroke events by heart disease background")

barplot(tbl[, 2], col = colors, main = "Confirmed stroke events by heart disease background",
        names.arg = c("Without heart disease", "With heart disease"))
hist(data$bmi, col = "royalblue", main = "BMI distribution", xlab = 'BMI')
hist(data$avg_glucose_level, col = "tomato", main = "Average glucose levels",
     xlab = "Average glucose levels")

data$age <- (data$age - mean(data$age)) / sd(data$age)
data$bmi <- (data$bmi - mean(data$bmi)) / sd(data$bmi)
data$avg_glucose_level <- (data$avg_glucose_level - mean(data$avg_glucose_level)) / sd(data$avg_glucose_level)

#Encoding

dummy <- dummyVars(" ~ . ", data = data)
data <- data.frame(predict(dummy, newdata = data))
table(data$stroke)

#Balancing the data

oversampled <- mwmote(data, classAttr = "stroke", numInstances = 500)
oversampled <- round(oversampled)

set.seed(1203)

fullData <- rbind(data, oversampled)

# Target class needs to be a factor
fullData$stroke <- factor(fullData$stroke)

#Dividing the data into train and test data into 80-20 ratio
sample <- createDataPartition(y = fullData$stroke, p = 0.8, list = FALSE)
train <- fullData[sample, ]
test <- fullData[-sample, ]

train_control <- trainControl(method = "cv", number = 5)


##Random Forest
library(randomForest)
#install.packages("cvms")
library(cvms)
rand.rf <- randomForest(as.factor(stroke) ~ ., data = train, ntree = 500, 
                          mtry = 4, nodesize = 5, importance = TRUE) 

rand.rf.pred <- predict(rand.rf, test)
ex <- confusionMatrix(rand.rf.pred, as.factor(test$stroke))
print(ex)
varImpPlot(rand.rf, type = 1)

fourfoldplot(ex$table, color = c("cyan", "pink"),conf.level = 0, margin = 1, main = "Confusion Matrix")
plot_confusion_matrix(rand.rf.pred)

##knn
knn <- train(stroke~., data = train, method = "knn", trControl = train_control)
knn
#install.packages("class")
library(class)
#knn.rf <- knn(train, test, train$stroke, k=3) 
                          
knn.rf.pred <- predict(knn, test)
# library(caret)
knnModel <- confusionMatrix(knn.rf.pred, as.factor(test$stroke))
print(knnModel)

fourfoldplot(knnModel$table, color = c("cyan", "pink"),conf.level = 0, margin = 1, main = "Confusion Matrix")
plot_confusion_matrix(knnModel)


###Logistic regression
logisticRegression <- train(stroke~., data = train, method = "glm", 
                            trControl = train_control,
                            family = "binomial")
logisticRegression
logit.reg <- glm(stroke ~ ., data = train, family = "binomial")
logit.reg.pred <- predict(logit.reg, test, type = "response")

logit.reg.pred.stroke <- ifelse(logit.reg.pred>0.5,1,0)
logist <- confusionMatrix(as.factor(logit.reg.pred.stroke),
                as.factor(test$stroke))
print(logist)
fourfoldplot(logist$table, color = c("cyan", "pink"),conf.level = 0, margin = 1, main = "Confusion Matrix")
plot_confusion_matrix(logist)

###Neural Networks
library(neuralnet)
library(nnet)
library(caret)

indx <- sapply(train, is.factor)
train[indx] <- lapply(train[indx], function(x) as.numeric(as.character(x)))

nn<-neuralnet(stroke ~ .,data=train,hidden = c(4,2),linear.output = FALSE)
plot(nn)
nn.pred <- predict(nn,test)

nnist <- confusionMatrix(nn.pred, as.factor(test$stroke))
nnist

# Ensemble using Weighted Average
# Taking weighted average of predictions
test$pred_weighted<-(as.numeric(rand.rf.pred)*0.25)+(as.numeric(logit.reg.pred)*0.25)
#Splitting into binary classes at 0.5
test$pred_weighted<-as.factor(ifelse(test$pred_weighted>0.5,'1','0'))
ense<-confusionMatrix(as.factor(test$stroke),as.factor(test$pred_weighted))
ense
fourfoldplot(ense$table, color = c("cyan", "pink"),conf.level = 0, margin = 1, main = "Confusion Matrix")
plot_confusion_matrix(ense)

