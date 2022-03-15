function netc = NN_classifier(trial)
    num_neurons = 98;
    num_direction = 8;
    num_trials = size(trial,1);
    
    spikes_num = [];%zeros(num_neurons,num_trials*num_direction);
    labels = [];
    
    for direc=1:num_direction
       for tr=1:num_trials
           dt = round(size(trial(tr,direc).spikes,2)/2);
           t = dt+1:dt:size(trial(tr,direc).spikes,2);
           for i=t
               spikes_num = [spikes_num, sum(trial(tr,direc).spikes(:,i-dt:i),2)];
%                spikes_num(:,tr*direc)=sum(trial(tr,direc).spikes(:,1:i),2);
               labels=[labels;direc];
           end
       end
    end

%     idxs = randperm(size(spikes_num,2));
%     spikes_num_ = spikes_num(:,idxs);
%     labels_=labels(idxs);
    target = ind2vec((labels).');
    netc = patternnet(5,'trainlm');
    netc.divideParam.trainRatio = 80/100;
    netc.divideParam.valRatio = 20/100;
    netc.divideParam.testRatio = 0/100;
    netc = train(netc,spikes_num,target);
end