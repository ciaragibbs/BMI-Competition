%% PCA+LDA Analysis i.e. Most Discriminant Feature Method
% Last update: 07/03/22 (incomplete)

load("monkeydata_training.mat")
noDirections = 8;
group = 10;
win = 50;
noTrain =  100;
trialProcess =  bin_and_sqrt(trial, group, 1);
trialFinal = get_firing_rates(trialProcess,group,win);
[trainData,testData] = split_test_train(trialFinal,noTrain);
%all_rates = combine_rates(train_data,500);
reachAngles = [30 70 110 150 190 230 310 350]; % given in degrees
trimmer = 500/group; % make the trajectories the same length
firingData = zeros([size(trainData(1,1).rates,1)*trimmer,noTrain*noDirections]);
noNeurons = size(trainData(1,1).rates,1);

% need to get (neurons x time)x trial
for i = 1: noDirections
    for j = 1: noTrain
        for k = 1: trimmer
            firingData(noNeurons*(k-1)+1:noNeurons*k,noTrain*(i-1)+j) = trainData(j,i).rates(:,k);     
        end
    end
end


% The aim of the next section is to identify the reaching direction
% associated with each trial of this monkey's session

% supervised labelling for Linear Discrminant Analysis
dirLabels = [1*ones(1,noTrain),2*ones(1,noTrain),3*ones(1,noTrain),4*ones(1,noTrain),5*ones(1,noTrain),6*ones(1,noTrain),7*ones(1,noTrain),8*ones(1,noTrain)];

% implement Principal Component Analysis 
[princComp,eVals]= getPCA(firingData);

%%
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

% with this need to optimise the Fisher's criterion - in particular the
% most discriminant feature method (i.e. PCA --> LDA to avoid issues with
% low trial: neuron ratios)
pcaDim = 200;
[eVectsLDA, eValsLDA] = eig(((princComp(:,1:pcaDim)'*scatWithin*princComp(:,1:pcaDim))^-1 )*(princComp(:,1:pcaDim)'*scatBetween*princComp(:,1:pcaDim)));
[~,sortIdx] = sort(diag(eValsLDA),'descend');
% optimum output
optimOut = princComp(:,1:pcaDim)*eVectsLDA(:,sortIdx(1:2));
% optimum projection from the Most Discriminant Feature Method....!
W = optimOut'*(firingData - mean(firingData,2));

%%
colors = {[1 0 0],[0 1 1],[1 1 0],[0 0 0],[0 0.75 0.75],[1 0 1],[0 1 0],[1 0.50 0.25]};
figure
hold on
for i=1:noDirections
    plot(W(1,noTrain*(i-1)+1:i*noTrain),W(2,noTrain*(i-1)+1:i*noTrain),'o','Color',colors{i},'MarkerFaceColor',colors{i},'MarkerEdgeColor','k')
    hold on
end

legend('30','70','110','150','190','230','310','350');
%reachAngles = [30 70 110 150 190 230 310 350];

%%
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