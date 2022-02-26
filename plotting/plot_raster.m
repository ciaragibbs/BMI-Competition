function [] = plot_raster(trial,tr,di, neuron_array)

% trial struct - data given
% tr - trial of interest
% di - direction of interest
% neuron_array - neurons you want to plot fot, should be a vector input

    figure 
    hold on
    time_axis = 1:size(trial(tr,di).spikes,2);
    for i = neuron_array
        time_spiking = find(trial(tr,di).spikes(i,:)==1);
        % where are the spikes at?
        for j = 1: length(time_spiking)
            plot([time_spiking(j) time_spiking(j)],[i-0.5 i+0.5], 'k');
            hold on
        end
    end
    xline(300,'b')
    xlabel('Time (ms)')
    ylabel('Neural Unit')
    axis tight
    title('Raster Plot')
    set(gcf,'color','w')

end