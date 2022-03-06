% Code for raw population decoding, assume there are neurons within the 98 that encode some angles and speed with their firing rates. Hence if we average the right neurons' firing rate it should be
% possible to predict the angle and speed . Requires functions in the preprocessing so please add them to the MATLAB path when running this file.

%% Test
clear; close all
load monkeydata_training.mat
[train0,test0] = split_test_train(trial,80);
train1 = bin_and_sqrt(train0,1,true);
train_rates = get_firing_rates(train1, 1, 40); % Assuming a Gaussian model?
% test_rates = get_firing_rate_rect(test1, 1, 40);
train_v = find_velocity(train_rates);

% crate = combine_rates(test_rates,500);
% [train_data,test_data] = split_test_train(trial,train_no);

%% format raw data
%Get dimensions, goal: have matrices of nneuron x (ntrial x nangle)
ntrial = size(train0,1);
nangle = size(train0,2);
nneuron = 98;
%Matrices to store data
rates = [];
speeds = [];
angles = [];
%Append matrices, concatenate all firing rates after 300ms in all trials for each neuron
for n = 1:nneuron
    speed = [];
    rate = [];
    angle = [];
    for i = 1:ntrial
        for j = 1:nangle
            rate = [rate, train_v(i,j).rates(n,301:end)];
            speed = [speed, train_v(i,j).speed(301:end)];
            angle = [angle, train_v(i,j).angle(301:end)];
        end
    end
    speeds = [speeds; speed];
    rates = [rates; rate];
    angles = [angles; angle];
end

%% plot the raw data (need to choose a neuron, here it's 96)
close all
figure
plot(std(rates,0,2,'omitnan')./mean(rates,2));
figure
scatter(angles(96,:),rates(96,:),'.')
figure
scatter(speeds(96,:),rates(96,:),'.')

%% tuning curve for reaching angle
% close all
angless = [30,70,110,150,190,230,310,350]; %Reaching angles
frs = zeros(nangle, ntrial); % Array for mean firing rate within a trial
figure
hold on
for cellid = 1:nneuron
    for i = 1:nangle
        color = ones(1,3)*0.25+rand(1,3)*0.75;
        for j = 1:ntrial
            frs(i,j) = mean(train_v(j,i).rates(cellid,301:end));
        end
    end
    %Take average and std over all trials for a neuron on one reaching angle and plot
    tun_fr = mean(frs,2); 
    err = std(frs,0,2);
    errorbar(angless, tun_fr, err)
end
hold off

%% movement angle tuning curve
ang_gap = 5; % in degree, the resolution of the tuning curve
angs = linspace(-pi,pi,360/ang_gap+1); %Define intervals of angles to be averaged
locs_a = cell(360/ang_gap,1); % Cell array to store indices where are the elements of the raw 'angles' array within each angle intervals, only 1D because the trajectory is the same for all neurons
tuning_angs = zeros(1,360/ang_gap); % Array to store mean angles of each angle interval 
tuning_rates_ang = zeros(360/ang_gap,nneuron); %Array to store mean firing rates of each neuron in each angle intervals
tuning_errs_a = zeros(360/ang_gap,nneuron); %Array to store std of firing rates of each neuron in each angle intervals
% Record the locations/indices of the elements within the different angle intervals and calculate the mean angle for each angle interval
for a = 1:360/ang_gap
    loc = find((angles(1,:) >= angs(a)) & (angles(1,:) < angs(a+1)));
    locs_a{a} = loc;
    tuning_angs(a) = mean(angles(1,loc)); % On a second thought maybe this should be calculated using vectors but maybe fine as angle intervals are small
end
% Calculate mean and std firing rates for all neurons at all angles
for n = 1:nneuron
    for a = 1:360/ang_gap
        tuning_rates_ang(a,n) = mean(rates(n,locs_a{a}));
        tuning_errs_a(a,n) = std(rates(n,locs_a{a}));
    end
end
% plot the tuning curves
figure
hold on
for n = 1:nneuron
    %     color = ones(1,3)*0.25+rand(1,3)*0.75;
    %     errorbar(tuning_angs, tuning_rates(:,n), tuning_errs(:,n))
    tuning_rates_ang(180/ang_gap+1,n) = mean(tuning_rates_ang(180/ang_gap-1:180/ang_gap+3,n));
    plot(tuning_angs,(tuning_rates_ang(:,n)))
end
hold off

%% movement speed tuning curve
%Similar to angle, just replacing the angle and tuning_rates_ang with speed related variables
speed_int = linspace(0,max(speed),50+1);
locs_s = cell(50,1);
tuning_speeds = zeros(1,50);
tuning_rates_s = zeros(50,nneuron);
tuning_errs_s = zeros(50,nneuron);
for s = 1:50
    loc = find((speeds(1,:) >= speed_int(s)) & (speeds(1,:) < speed_int(s+1)));
    locs_s{s} = loc;
    tuning_speeds(s) = mean(speeds(1,loc));
end
for n = 1:nneuron
    for s = 1:50
        tuning_rates_s(s,n) = mean(rates(n,locs_s{s}));
        tuning_errs_s(s,n) = std(rates(n,locs_s{s}));
    end
end
figure
hold on
for n = 1:nneuron
    %     color = ones(1,3)*0.25+rand(1,3)*0.75;
    %     errorbar(tuning_angs, tuning_rates(:,n), tuning_errs(:,n))
    plot(tuning_speeds, tuning_rates_s(:,n))
end
hold off

%% extract info
step_size = mean(speed); % Not sure how to predict speed best yet, so just using mean atm (INACCURATE as shown later)
usefuls = []; %Array to store neurons that show enough variability in firing rate
%Arrays to store weights, atm all weights = 1, _p, _t denote peak and trough firing rate
weights_p = []; 
weights_t = [];
% Arrays to store angles at which the neuron has max/min firing rates
peak_a = [];
trough_a = [];
% Arrays to store neurons' max/min firing rates
peak_f = [];
trough_f = [];
%Array to store std of neurons' firing rates
stds = [];
for n = 1:nneuron
    %Calculate the range and range relative to the std for each neuron
    range = max(tuning_rates_ang(:,n)) - min(tuning_rates_ang(:,n));
    zrange = max(zscore(tuning_rates_ang(:,n))) - min(zscore(tuning_rates_ang(:,n)));
    %Use arbitrary threshold to select eneurons that has enough variations, in future may use alternative criteria
    if range >= 3 && zrange >= 1 % && std(tuning_rates_ang(:,n)) >= 3
        %Find the max/min firing rates and their corresponding angle (can be found by using this ang_p/_t index in the tuning_angs array)
        [freq_p,ang_p] = max(tuning_rates_ang(:,n));
        [freq_t,ang_t] = min(tuning_rates_ang(:,n));
        % Check if these angles have already been recorded in the recording array, as we use equal weights, multiple presence of certain angles can drift the population decoding result towards that
        % angle and introduce a bias
        exist_p = find(peak_a == tuning_angs(ang_p));
        exist_t = find(trough_a == tuning_angs(ang_t));
        if length(usefuls) == 0 %If nothing has been recorded, just record whatever is found
            peak_a = [peak_a, tuning_angs(ang_p)];
            peak_f = [peak_f, freq_p];
            trough_a = [trough_a, tuning_angs(ang_t)];
            trough_f = [trough_f, freq_t];
            usefuls = [usefuls, n];
            weights_p = [weights_p, 1];
            weights_t = [weights_t, 1];
            stds = [stds, std(tuning_rates_ang(:,n))];
        elseif sum(exist_p) == 0 && sum(exist_t) == 0 %If this neuron has both peak and trough that are never recorded then record this neuron
            peak_a = [peak_a, tuning_angs(ang_p)];
            peak_f = [peak_f, freq_p];
            trough_a = [trough_a, tuning_angs(ang_t)];
            trough_f = [trough_f, freq_t];
            usefuls = [usefuls, n];
            weights_p = [weights_p, 1];
            weights_t = [weights_t, 1];
            stds = [stds, std(tuning_rates_ang(:,n))];
            % There may be a better way to include more neurons without introducing bias but probably needs to tune weights
        end
    end
end
% Plot the selected population's angle coverage
figure
hold on
plot(trough_f,trough_a,'.')
plot(peak_f,peak_a,'.')

%% real test
%Weight experiments, ngl for now
% weights_p = ones(1, length(weights_p));
% weights_t = ones(1, length(weights_t));
% weights_p = 1./weights_p;
% weights_t = 1./weights_t;

% Load and process the test data
test2 = bin_and_sqrt(test0,1,true);
exam = get_firing_rates(test2, 1, 40);
figure

trialid = 1; %Select the trial for comparison
for ori = 1:8 %For all reaching angles
    subplot(2,4,ori)
    trace1 = exam(trialid,ori).rates(:,301:end); %Get firing rates for all neurons. Start after 300ms as not much happens before stimulus given
    traceu = trace1(usefuls,:); %Select the firing rates of the variable enough neurons identified in training
    n_use = length(usefuls);%Get number of neurons for loop
    d0 = exam(trialid,ori).handPos(:,301); % initial position vector
    d = [d0]; %Array for trajectory
    for t = 1:size(traceu,2)-1% time steps
        dirs = []; %Array to store direction vectors for each neuron used
        for n = 1:n_use% Loop through neurons
            if traceu(n,t) >= (peak_f(n)-stds(n) ) %If firing rate close to peak then use the angle corresponding to the peak for direction
                dirs = [dirs, weights_p(n)*[cos(peak_a(n));sin(peak_a(n))]]; %Convert angle into unit vector in 2D
            elseif traceu(n,t) <= trough_f(n)+stds(n) %If firing rate close to trough then use the angle corresponding to the trough for direction
                dirs = [dirs, weights_t(n)*[cos(trough_a(n));sin(trough_a(n))]];
            end
        end
        dir = sum(dirs,2); % Vector sum
        dir = dir/norm(dir); %Normalise to unit vector
        if length(dir) ~= 0 % In case no neuron was active/inactive enough assume no movements, otherwise increment the trajectory with the mean vector
            d0 = d0 + step_size*(dir);
        end
        d = [d, d0]; %Update the trajectory
    end
    %Plot the true and predicted trajectory
    hold on
    plot(exam(trialid,ori).handPos(1,301:end),exam(trialid,ori).handPos(2,301:end),'.')
    plot(d(1,:),d(2,:),'.'); 
    hold off
end
