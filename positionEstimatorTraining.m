function [modelParameters]=positionEstimatorTraining(trial)
    spikes = [];
    vels = [];
    num_neurons = 98;
    % Cocatenate all the spikes from each trial for each neuron (direction 1 only)
    for n=1:num_neurons
        tmp = [];
        for tr=1:length(trial)
            tmp =[tmp trial(tr,1).spikes(n,:)];
        end
        spikes(n,:) = tmp;
    end
    
    % Concatenate the velocities from all trials (direction 1 only)
    for tr=1:length(trial)
       vel = find_velocity(trial(tr,1));
       vels = [vels vel.velocity(1:2,:)];
    end
    
    dt = 5; %Batch size
    % Instead of using the spikes as input to the NN, we take use the firing
    % rate of each neuron within a batch of dt ms
    neural_data = preprocess_input(spikes,dt); 
    output_binned = preprocess_output(vels,dt); %Output is separated in batches as well

    % Create NN
    sizeHidden = [20]; %num of neuron in each hidden Layer
    net = fitnet(sizeHidden,'trainlm');
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 0/100;

    % Train NN
    [net, ~] = train(net, neural_data', output_binned');
    
    modelParameters = net;

end