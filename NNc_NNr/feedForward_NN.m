% Feedforward NN
clear all; close all
load monkeydata_training.mat
spikes = [];
vels = [];

% Cocatenate all the spikes from each trial for each neuron (direction 1 only)
for n=1:98
    tmp = [];
    for tr=1:50
        tmp =[tmp trial(tr,1).spikes(n,:)];
    end
    spikes(n,:) = tmp;
end

% Concatenate the velocities from all trials (direction 1 only)
for tr=1:50
   vel = find_velocity(trial(tr,1));
   vels = [vels vel.velocity(1:2,:)];
end

dt = 5; %Batch size
% Instead of using the spikes as input to the NN, we take use the firing
% rate of each neuron within a batch of dt ms
neural_data = preprocess_input(spikes,dt); 
output_binned = preprocess_output(vels,dt); %Output is separated in batches as well

% Create NN
sizeHidden = [30]; %num of neuron in each hidden Layer
net = fitnet(sizeHidden,'trainlm');
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100;

% Train NN
[net, ~] = train(net, neural_data', output_binned');

% Example velocity and x-y trajectory prediction using trial 90 and direction 1
input_val = preprocess_input(trial(90,1).spikes,dt);
t = 0:dt:size(trial(90,1).spikes,2);

pred = net(input_val');

interp_predx = interp1(t(1:end-1),pred(1,:),0:size(trial(90,1).spikes,2),'linear');
interp_predy = interp1(t(1:end-1),pred(2,:),0:size(trial(90,1).spikes,2),'linear');

x_traj = cumsum([trial(90,1).handPos(1,1) interp_predx]);
y_traj = cumsum([trial(90,1).handPos(2,1) interp_predy]);
figure;
plot(x_traj,y_traj)
hold on;
plot(trial(90,1).handPos(1,:),trial(90,1).handPos(2,:))
