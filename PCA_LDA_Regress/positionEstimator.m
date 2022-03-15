function [x,y]= positionEstimator(past_current_trial, modelParameters)

% Last edit: 14/03/22
% Authors: Ciara Gibbs, Fabio Oliva, Yinzhe Wu, Zhiyu Zheng
% TO BE COMPLETED

% Inputs:
% testData: struct with very similar formatting to trial, other than it has
% the additinal field of starting position
% modelParameters: previously saved modelParameters from PCA-LDA analysis
% using the training dataset

% Outputs:
% decodedPosX: predicted X position according to the PCR model
% decodedPosY: predicted Y position according to the PCR model
% newParameters: any modifications in classification etc stored here

            noDirections = 8;
            group = 20;
            win = 50;
            trialProcess =  bin_and_sqrt(past_current_trial, group, 1); % preprocessing
            trialFinal = get_firing_rates(trialProcess,group,win); % preprocessing (including smoothing)
            reachAngles = [30 70 110 150 190 230 310 350]; % given in degrees
            T_end = size(past_current_trial.spikes,2)
            noNeurons = size(trialFinal(1,1).rates,1);

            % need to get (neurons x time)x trial for use with PCA and PCR
            firingData = reshape(trialFinal.rates, [], 1);

            % get the relevant parameters from the model
            if T_end >=320 && T_end <=560

                indexer =  (T_end/group)-(320/group)+1;
                WTrain = modelParameters.classify(indexer).wLDA_kNN;
                pcaDim = modelParameters.classify(indexer).dPCA_kNN;
                ldaDim = modelParameters.classify(indexer).dLDA_kNN;
                optimTrain = modelParameters.classify(indexer).wOpt_kNN;
                meanFiringTrain = modelParameters.classify(indexer).mFire_kNN;
                meanClusterTrain = modelParameters.classify(indexer).mCluster_kNN;
                % not sure whether it should be the mean from train or test
                WTest = optimTrain'*(firingData-meanFiringTrain); 
                outLabel = getKNNs(WTest, WTrain,ldaDim,8);

            elseif T_end >560 % i.e. just keep using the parameters derived with the largest length of training time
 
                indexer =  13;
                WTrain = modelParameters.classify(indexer).wLDA_kNN;
                pcaDim = modelParameters.classify(indexer).dPCA_kNN;
                ldaDim = modelParameters.classify(indexer).dLDA_kNN;
                optimTrain = modelParameters.classify(indexer).wOpt_kNN;
                meanFiringTrain = modelParameters.classify(indexer).mFire_kNN;
                meanClusterTrain = modelParameters.classify(indexer).mCluster_kNN;
                % not sure whether it should be the mean from train or test
                size(firingData)
                size(meanFiringTrain)
                WTest = optimTrain'*(firingData(1:2744)-meanFiringTrain); 
                outLabel = getKNNs(WTest, WTrain,ldaDim,8);


            end


            if T_end >=320 && T_end <=560
                
                indexer =  (T_end/group)-(320/group)+1;
                avX = modelParameters.averages(indexer).avX(:,outLabel);
                avY = modelParameters.averages(indexer).avY(:,outLabel);
                meanFiring = modelParameters.pcr(outLabel,indexer).fMean;
                bx = modelParameters.pcr(outLabel,indexer).bx;
                by = modelParameters.pcr(outLabel,indexer).by;
                x = (firingData - mean(meanFiring))'*bx + avX;
                y = (firingData - mean(meanFiring))'*by + avY;
                try
                    x =  x(T_end,1);
                    y = y(T_end,1);
                catch
                     x =  x(end,1);
                    y = y(end,1);
                end

            elseif T_end >560 % i.e. just keep using the model with the largest length of training time

                avX = modelParameters.averages(13).avX(:,outLabel);
                avY = modelParameters.averages(13).avY(:,outLabel);
                meanFiring = modelParameters.pcr(outLabel,13).fMean;
                bx = modelParameters.pcr(outLabel,13).bx;
                by = modelParameters.pcr(outLabel,13).by;
                x = (firingData(1:length(bx)) - mean(meanFiring))'*bx + avX;
                y = (firingData(1:length(by)) - mean(meanFiring))'*by + avY;
                try
                    x =  x(T_end,1);
                    y = y(T_end,1);
                catch
                     x =  x(end,1);
                    y = y(end,1);
                end
            end

            
            


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


end