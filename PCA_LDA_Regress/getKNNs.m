function [outLabels] = getKNNs(WTest, WTrain,dimLDA,nearFactor)


    % Inputs:
    % testingData: dimLDA x no. test trials, corresponding to the
    % projection of the trial data after use of PCA-LDA
    % trainingData: dimLDA x no. training trials, corresponding to the
    % projection of the trial data after use of PCA-LDA
    % dimLDA: model parameter that is used in positionEstimatorTraining to
    % fix how many dimensions we are use for clustering analysis
    % nearFactor: taking 1/nearFactor of each directions worth of data as
    % nearest neighbours
    
    % Outputs:
    % labels: reaching angle/direction labels of the testing data deduced with the kNN
    % algorithm
    
    trainMat = WTrain';
    testMat = WTest;
    trainSq = sum(trainMat.*trainMat,2);
    testSq = sum(testMat.*testMat,1);
    % allDists has dimensions no. test points x no. train points
    % i.e for every test point down a column, it's distance from every
    % training point is across a row
    allDists = trainSq(:,ones(1,length(testMat))) + testSq(ones(1,length(trainMat)),:) - 2*trainMat*testMat;
    allDists = allDists';


    % sort for the k nearest neigbours
    k = round(length(WTrain)/(8*nearFactor)); % i.e. comparing to 1/4 of the amount of trials for one direction
    [~,sorted] = sort(allDists,2);
    nearest = sorted(:,1:k);

    % what is the mode direction for these k-nearest neighbours?
    noTrain = size(WTrain,2)/8;
    dirLabels = [1*ones(1,noTrain),2*ones(1,noTrain),3*ones(1,noTrain),4*ones(1,noTrain),5*ones(1,noTrain),6*ones(1,noTrain),7*ones(1,noTrain),8*ones(1,noTrain)]';
    nearestLabs =  reshape(dirLabels(nearest),[],k);
    outLabels =  mode(nearestLabs,2);

end
