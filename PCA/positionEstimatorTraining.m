function  [modelParameters,princComp,W] = positionEstimatorTraining(trainingData)

% Last edit: 14/03/22
% Authors: Ciara Gibbs, Fabio Oliva, Yinzhe Wu, Zhiyu Zheng
% TO BE COMPLETED

noDirections = 8;
group = 20;
win = 50;
noTrain =  length(trainingData); % because it is set in the testFunction_for_students_MTb file
trialProcess =  bin_and_sqrt(trainingData, group, 1); % preprocessing
trialFinal = get_firing_rates(trialProcess,group,win); % preprocessing (including smoothing)
reachAngles = [30 70 110 150 190 230 310 350]; % given in degrees
trimmer = 500/group; % make the trajectories the same length
firingData = zeros([size(trialFinal(1,1).rates,1)*trimmer,noTrain*noDirections]);
noNeurons = size(trialFinal(1,1).rates,1);

startTime = 320;
endTime = 560;
trimmer =  endTime/group;
noIts = 1;% need to calculate the mean RMSE and std RMSE across iterations e.g. 10

modelParameters = struct;

% need to get (neurons x time)x trial
for i = 1: noDirections
    for j = 1: noTrain
        for k = 1: trimmer
            firingData(noNeurons*(k-1)+1:noNeurons*k,noTrain*(i-1)+j) = trialFinal(j,i).rates(:,k);     
        end
    end
end


% The aim of the next section is to identify the reaching direction
% associated with each trial of this monkey's session

% supervised labelling for Linear Discrminant Analysis
dirLabels = [1*ones(1,noTrain),2*ones(1,noTrain),3*ones(1,noTrain),4*ones(1,noTrain),5*ones(1,noTrain),6*ones(1,noTrain),7*ones(1,noTrain),8*ones(1,noTrain)];

% implement Principal Component Analysis 
[princComp,eVals]= getPCA(firingData);


% https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/old_IDAPILecture15.pdf

% use Linear Discriminant Analysis on Reduced Dimension Firing Data
matBetween = zeros(size(firingData,1),noDirections);
% for LDA need to get the across-class and within-class scatter matrices
for i = 1: noDirections
    matBetween(:,i) =  mean(firingData(:,noTrain*(i-1)+1:i*noTrain),2);
end
scatBetween = (matBetween - mean(firingData,2))*(matBetween - mean(firingData,2))';
scatGrand =  (firingData - mean(firingData,2))*(firingData - mean(firingData,2))';
scatWithin = scatGrand - scatBetween; % as per pdf above
%^^ double checking - done


pcaDim = 50; % just arbitrary atm... will need to look at the eigenvalues to get a better estimate!
% for this to work, a regression model needs to be constructed for each
% angle, for a given number of PCA dimensions
ldaDim = 3; % also arbitrary atm...

% with this need to optimise the Fisher's criterion - in particular the
% most discriminant feature method (i.e. PCA --> LDA to avoid issues with
% low trial: neuron ratios)
[eVectsLDA, eValsLDA] = eig(((princComp(:,1:pcaDim)'*scatWithin*princComp(:,1:pcaDim))^-1 )*(princComp(:,1:pcaDim)'*scatBetween*princComp(:,1:pcaDim)));
[~,sortIdx] = sort(diag(eValsLDA),'descend');
% optimum output
optimOut = princComp(:,1:pcaDim)*eVectsLDA(:,sortIdx(1:3));
% optimum projection from the Most Discriminant Feature Method....!
W = optimOut'*(firingData - mean(firingData,2));


% for now just going to try with a kNN classifier when testing, but will
% see if another classifier ends up better later...

% store these weightings in the model parameter struct
modelParameters.wLDA_kNN = W;
modelParameters.dPCA_kNN = pcaDim;
modelParameters.dLDA_kNN = ldaDim;

% perform Principal Component Regression

for it = 1:noIts
   
    firingData = zeros([size(trainData(1,1).rates,1)*trimmer,noTrain*noDirections]);
    noNeurons = size(trainData(1,1).rates,1);
    
    % shaping of the firing rate data for PCA-based decomposition (SVD)
    for i = 1: noDirections
        for j = 1: noTrain
            for k = 1: trimmer
                firingData(noNeurons*(k-1)+1:noNeurons*k,noTrain*(i-1)+j) = trainData(j,i).rates(:,k);     
            end
        end
    end
    [xn,yn,xrs,yrs] = getEqualandSampled(trainData,noDirections,noTrain,group);
    % only take the indices that correspond to the testing intervals
    xTestInt = xrs(:,[startTime:group:endTime]/20,:);
    yTestInt = yrs(:,[startTime:group:endTime]/20,:);
    
    
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
            trialProcessed(j,i).handPos = trial(j,i).handPos(1:2,:);
            trialProcessed(j,i).bin_size = group; % recorded in ms
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
            trialFinal(j,i).handPos = trialProcessed(j,i).handPos;
            trialFinal(j,i).bin_size = trialProcessed(j,i).bin_size; % recorded in ms
        end
    end

end


function [prinComp,evals,sortIdx,ev]= getPCA(data)
    % subtract the cross-trial mean
    dataCT = data - mean(data,2);
    % calculate the covariance matrix
    covMat = dataCT'*dataCT/size(data,2);
    % get eigenvalues and eigenvectors
    [evects, evals] = eig(covMat);
    % sort both eigenvalues and eigenvectors from largest to smallest weighting
    [~,sortIdx] = sort(diag(evals),'descend');
    evects = evects(:,sortIdx);
    % project firing rate data onto the newly derived basis
    prinComp = dataCT*evects;
    % normalisation
    prinComp = prinComp./sqrt(sum(prinComp.^2));
    % just getting the eigenvalues and not all the other zeros of the diagonal
    % matrix
    evalsDiag = diag(evals);
    evals = diag(evalsDiag(sortIdx));
end




end