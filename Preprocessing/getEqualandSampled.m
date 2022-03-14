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