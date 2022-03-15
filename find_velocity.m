function trial_with_v = find_velocity(trial)
% This function calculates the 'velocity' (i.e. position difference between neighbouring time steps) for a dataset struct, it will return 
% a new struct with a velocity attribute (? not sure about the terminology but basically trial.velocity). Moreover, it will also calculate
% the direction in x-y plane (angle) and the magnitude (speed)
trial_with_v = trial;

for i = 1: size(trial,1)

        for j = 1:size(trial,2)
            % Calculate difference between every timestep's handpos, atm the velocity for the last step will be 0, can choose to ignore
            trial_with_v(i,j).velocity = [diff(trial(i,j).handPos,1,2), zeros(size(trial(i,j).handPos,1),1)]; 
            trial_with_v(i,j).angle = atan2(trial_with_v(i,j).velocity(2,:),trial_with_v(i,j).velocity(1,:));
            trial_with_v(i,j).speed = sqrt(sum(trial_with_v(i,j).velocity.^2, 1));
        end
        
end

end