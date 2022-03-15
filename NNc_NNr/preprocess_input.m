function neural_data = preprocess_input(spikes,dt)
%     spikes = trial(1,1).spikes;
    spike_times = cell(size(spikes,1),1);
    for i=1:size(spikes,1)
       t_fire=1; 
       for j=1:size(spikes,2)
          if spikes(i,j)==1
             spike_times{i}= [spike_times{i},t_fire];
          end
          t_fire=t_fire+1;
       end
    end

%     dt = 20;
    t_start = 0;
    t_end = size(spikes,2);

    edges = t_start:dt:t_end;
    num_bins = size(edges,2)-1;
    num_neurons = size(spike_times,1);
    neural_data = zeros(num_bins,num_neurons);

    for i=1:num_neurons
       neural_data(:,i)= histcounts(spike_times{i},edges);
    end

end