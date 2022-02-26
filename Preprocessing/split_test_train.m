function [train_data,test_data] = split_test_train(trial,train_no)

% this function creates a random segragation of data for training and
% testing, according to the number of training samples you want

% trial =  struct of the data
% train_no =  number of training samples

    tot =  size(trial,1);
    splitter = randi([0, tot-1], 1,tot) + (1:tot:tot);
    train_ind= splitter(1:train_no);
    test_ind = splitter(train_no+1:end);
    
    train_data = trial(train_ind);
    test_data = trial(test_ind);

end