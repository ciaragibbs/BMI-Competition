function output_processed = preprocess_output(handPos,dt)
    t_start = 0;
    t_end = size(handPos,2);
    edges = t_start:dt:t_end;
    num_bins = size(edges,2)-1;
    output_dim = size(handPos,1);
    
    output_processed = zeros(num_bins,output_dim);
    output_times = [t_start:t_end]';
    outputs = handPos';
    for i=1:num_bins
       idxs = find((output_times>=edges(i)) & (output_times<edges(i+1)));
       for j=1:output_dim
          output_processed(i,j) = mean(outputs(idxs,j)); 
       end
    end
    
end
