function concatRates =  combine_rates(trial,limit)


% stacks all of the rate/spike data so that it can given easily as input
% into PCA

% limit is the data we are using up to, but you also need to know the
% binning resolution for this... provide it in 1ms resolution, and code
% will convert

    group = trial(1,1).bin_size;
    limit = limit/group;
    concatRates = [];
    
    for i = 1: size(trial,2)
        for j = 1: size(trial,1)
            
            concatRates = [concatRates; trial(j,i).rates(:,1:limit)'];
            
        end
    end
    
end