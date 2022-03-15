% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

load("monkeydata_training.mat")

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:80),:);
testData = trial(ix(81:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            t
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            

            [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
        
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions);



function trialProcessed = bin_and_sqrt(trial, group, to_sqrt)

% Use to re-bin to different resolutions and to sqrt binned spikes (is used
% to reduce the effects of any significantly higher-firing neurons, which
% could bias dimensionality reduction)

% trial = the given struct
% group = new binning resolution - note the current resolution is 1ms
% to_sqrt = binary , 1 -> sqrt spikes, 0 -> leave

    trialProcessed = struct;
    

    for i = 1: size(trial,2)
        for j = 1: size(trial,1)

            all_spikes = trial(j,i).spikes; % spikes is no neurons x no time points
            no_neurons = size(all_spikes,1);
            no_points = size(all_spikes,2);
            t_new = 1: group : no_points +1; % because it might not round add a 1 
            spikes = zeros(no_neurons,numel(t_new)-1);

            for k = 1 : numel(t_new) - 1 % get rid of the paddded bin
                spikes(:,k) = sum(all_spikes(:,t_new(k):t_new(k+1)-1),2);
            end

            if to_sqrt
                spikes = sqrt(spikes);
            end

            trialProcessed(j,i).spikes = spikes;
%             trialProcessed(j,i).handPos = trial(j,i).handPos(1:2,:);
%             trialProcessed(j,i).bin_size = group; % recorded in ms
        end
    end
    
end


function trialFinal = get_firing_rates(trialProcessed,group,scale_window)

% trial = struct , preferably the struct which has been appropaitely binned
% and had low-firing neurons removed if needed
% group = binning resolution - depends on whether you have changed it with
% the bin_and_sqrt function
% scale_window = a scaling parameter for the Gaussian kernel - am
% setting at 50 now but feel free to mess around with it

    trialFinal = struct;
    win = 10*(scale_window/group);
    normstd = scale_window/group;
    alpha = (win-1)/(2*normstd);
    gaussian_window = gausswin(win,alpha)/sum(gausswin(win,alpha));
    
    for i = 1: size(trialProcessed,2)

        for j = 1:size(trialProcessed,1)
            
            hold_rates = zeros(size(trialProcessed(j,i).spikes,1),size(trialProcessed(j,i).spikes,2));
            
            for k = 1: size(trialProcessed(j,i).spikes,1)
                
                hold_rates(k,:) = conv(trialProcessed(j,i).spikes(k,:),gaussian_window,'same')/(group/1000);
            end
            
            trialFinal(j,i).rates = hold_rates;
%             trialFinal(j,i).handPos = trialProcessed(j,i).handPos;
%             trialFinal(j,i).bin_size = trialProcessed(j,i).bin_size; % recorded in ms
        end
    end

end


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