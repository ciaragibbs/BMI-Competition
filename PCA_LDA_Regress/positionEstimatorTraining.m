function  [modelParameters,princComp,W] = positionEstimatorTraining(trainingData)

% Last edit: 14/03/22
% Authors: Ciara Gibbs, Fabio Oliva, Yinzhe Wu, Zhiyu Zheng
% think it's completed...
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
ldaDim = 2; % also arbitrary atm...

% with this need to optimise the Fisher's criterion - in particular the
% most discriminant feature method (i.e. PCA --> LDA to avoid issues with
% low trial: neuron ratios)
[eVectsLDA, eValsLDA] = eig(((princComp(:,1:pcaDim)'*scatWithin*princComp(:,1:pcaDim))^-1 )*(princComp(:,1:pcaDim)'*scatBetween*princComp(:,1:pcaDim)));
[~,sortIdx] = sort(diag(eValsLDA),'descend');
% optimum output
optimOut = princComp(:,1:pcaDim)*eVectsLDA(:,sortIdx(1:ldaDim));
% optimum projection from the Most Discriminant Feature Method....!
W = optimOut'*(firingData - mean(firingData,2));


% for now just going to try with a kNN classifier when testing, but will
% see if another classifier ends up better later...

% store these weightings in the model parameter struct
modelParameters.wLDA_kNN = W;
modelParameters.dPCA_kNN = pcaDim;
modelParameters.dLDA_kNN = ldaDim;
modelParameters.wOpt_kNN = optimOut;
modelParameters.mFire_kNN = mean(firingData,2);

% get the mean of the clusters according to the label
meanClusters = [];
for i = 1: noDirections
    meanClusters = [meanClusters, mean(W(:,dirLabels == i))];
end

modelParameters.mCluster_kNN = meanClusters;

%PCR 
% https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Principal_Components_Regression.pdf

% using firingData that has already been calculated above
[xn,yn,xrs,yrs] = getEqualandSampled(trainingData,noDirections,noTrain,group);
% only take the indices that correspond to the testing intervals
xTestInt = xrs(:,[startTime:group:endTime]/20,:);
yTestInt = yrs(:,[startTime:group:endTime]/20,:);

%
% PCA/SVD

timeDivision = repelem(group:group:endTime,noNeurons);
testingTimes = startTime:group:endTime;
% double check - done

for i = 1: noDirections
    
    getXdirection = squeeze(xTestInt(:,:,i));
    getYdirection = squeeze(yTestInt(:,:,i));
    
    for j = 1: ((endTime-startTime)/group)+1
        
        x4pcr = getXdirection(:,j) - mean(getXdirection(:,j));
        y4pcr = getYdirection(:,j) - mean(getYdirection(:,j));
        
        windowedFiring = firingData(timeDivision <= testingTimes(j),dirLabels == i);
        [eVects,~]= getPCA(windowedFiring);
        % project data onto principle components
        % arbitrary choice of no. of principal components used
        Z = eVects(:,1:pcaDim)'*(windowedFiring - mean(windowedFiring,1));
        % by pdf:
        % B = PA
        % P is the eigenvector matrix
        % A is the an estimation formula, based on the principal component
        % data matrix Z
        % B is the regression coefficients,for the x and y positions
        
        Bx = (eVects(:,1:pcaDim)*inv(Z*Z')*Z)*x4pcr;
        By = (eVects(:,1:pcaDim)*inv(Z*Z')*Z)*y4pcr;
        
        modelParameters.pcr(i,j).bx = Bx;
        modelParameters.pcr(i,j).by = By;
        
    end
    
    modelParameters.avX = squeeze(mean(xn,1));
    modelParameters.avY = squeeze(mean(yn,1));
    
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

function [xn,yn,xrs,yrs] = getEqualandSampled(data,noDirections,noTrain,group)

% Inputs:
% data is the struct of data = spike or rate form, doesn't make a
% difference
% noDirections = number of different reaching angle directions i.e. 8
% noTrain = number of training samples being used 
%(note will need to add an option for train or test maybe, or maybe not
%important...)
% group = binning resolution of the spiking/rate data

% given a change in the binning resolution of the spiking data,
% accordingly, the position informatin must be downsampled

% also all the trajectories need to be the same length, so to do this we
% will continuously add the end value onto the trajectories that are less
% than the max length


% Find the maximum trajectory
trialHolder = struct2cell(data);
sizes = [];
for i = [2:3: noTrain*noDirections*3]
    sizes = [sizes, size(trialHolder{i},2)];
end
maxSize = max(sizes);
clear trialHolder

% preallocate for position matrices
xn = zeros([noTrain, maxSize, noDirections]);
yn = xn;
% preallocate for resampled position matrices

% padding position data with the end value
for i = 1: noDirections
    for j = 1:noTrain

        xn(j,:,i) = [data(j,i).handPos(1,:),data(j,i).handPos(1,end)*ones(1,maxSize-sizes(noTrain*(i-1) + j))];
        yn(j,:,i) = [data(j,i).handPos(2,:),data(j,i).handPos(2,end)*ones(1,maxSize-sizes(noTrain*(i-1) + j))];  
        % resampling according to the binning size
        tempx = xn(j,:,i);
        tempy = xn(j,:,i);
        xrs(j,:,i) = tempx(1:group:end);
        yrs(j,:,i) = tempy(1:group:end);
    
    end
end

end



end