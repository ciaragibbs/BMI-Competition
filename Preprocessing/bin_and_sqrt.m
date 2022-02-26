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