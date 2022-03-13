clear
load("monkeydata_training.mat")
train_no = 60;
[trainData, testData] = split_test_train(trial,train_no);
[modelParameters,princComp,W] = positionEstimatorTraining(trainData);
outLabels= positionEstimator(testData, modelParameters);

