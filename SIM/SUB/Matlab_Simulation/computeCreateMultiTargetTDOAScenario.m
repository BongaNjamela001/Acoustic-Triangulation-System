function [scenario, tdoaPairs, receiverIDs] = computeCreateMultiTargetTDOAScenario(numReceivers, numTgts)

% This function defines the multi-target scenario used in the TDOA tracking
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

% Add targets
S = rng(1);
xTarget = -5000 + 10000*rand(1,numTgts);
yTarget = -5000 + 10000*rand(1,numTgts);
vxTarget = 50*randn(1,numTgts);
vyTarget = 50*randn(1,numTgts);
zTarget = zeros(1,numTgts);
vzTarget = zeros(1,numTgts);
rng(S);

for i = 1:numTgts
target = platform(scenario);
target.Trajectory.Position = [xTarget(i) yTarget(i) zTarget(i)];
target.Trajectory.Velocity = [vxTarget(i) vyTarget(i) vzTarget(i)];
end

% PlatformIDs of TDOA calculation pairs. Each row represents the TDOA pair
% [1 3] means a TDOA is calculated between 1 and 3 with 3 as the reference
% receiver.
tdoaPairs = (1:(numReceivers-1))';
tdoaPairs(:,2) = numReceivers;

% IDs of all receivers
receiverIDs = 1:numReceivers;

end
