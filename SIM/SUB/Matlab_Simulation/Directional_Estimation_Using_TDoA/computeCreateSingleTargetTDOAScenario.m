function [scenario, tdoaPairs, receiverIds] = computeCreateSingleTargetTDOAScenario(numReceivers)

% This function defines the single-target scenario used in the TDOA tracking
% example. 

% Copyright 2021 The MathWorks, Inc.

scenario = trackingScenario('StopTime',60);
scenario.UpdateRate = 1;

theta = linspace(-pi,pi,numReceivers+1);
r = 5000;
xReceiver = r*cos(theta(1:end-1));
yReceiver = r*sin(theta(1:end-1));

% The algorithm shown in the example is suitable for 3-D workflows. Each
% receiver must be at different height to observe/estimate the z of
% the object. 
zReceiver = zeros(1,numReceivers);

for i = 1:numel(xReceiver)
    p = platform(scenario);
    p.Trajectory.Position = [xReceiver(i) yReceiver(i) zReceiver(i)];
end

% Add target
target = platform(scenario);
target.Trajectory.Position = [-2500 2000 1500*(numReceivers >= 4)];
target.Trajectory.Velocity = [150 0 0];

% PlatformIDs of TDOA calculation pairs. Each row represents the TDOA pair
% [1 3] means a TDOA is calculated between 1 and 3 with 3 as the reference
% receiver.
tdoaPairs = (1:(numReceivers-1))';
tdoaPairs(:,2) = numReceivers;

receiverIds = 1:numReceivers;
end
