function [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters)
    dt = 5;
    input_val = preprocess_input(past_current_trial.spikes,dt);
    t = 0:dt:size(past_current_trial.spikes,2);

    pred = modelParameters(input_val');

    interp_predx = interp1(t(1:end-1),pred(1,:),0:size(past_current_trial.spikes,2),'linear');
    interp_predy = interp1(t(1:end-1),pred(2,:),0:size(past_current_trial.spikes,2),'linear');

    x_traj = cumsum([past_current_trial.startHandPos(1) interp_predx]);
    y_traj = cumsum([past_current_trial.startHandPos(2) interp_predy]);
    x_traj = rmmissing(x_traj);
    y_traj = rmmissing(y_traj);
%     plot(x_traj,y_traj)
%     display([x_traj,y_traj])
    decodedPosX = x_traj(end);
    decodedPosY = y_traj(end);
end
