function detections = computeSimulateTOA(scenario, receiverIDs, measNoise, Pd, nFalse)

% This function simulates TOA for each receiver
%
% Define the detection format. The detection is a objectDetection object.
% Each property of the objectDetection object is defined as
% 
% SensorIndex - Unique identifier of receiver. 
% Time - Current elapsed simulation time (s)
% Measurement - TOA measurement of target (ns)
% MeasurementNoise - uncertainty variance in TOA measurement (ns^2)
% MeasurementParameters - A struct defining the position of
% receiver in the scenario as field OriginPosition 

nReceivers = numel(receiverIDs);

% Find targets
platIDs = cellfun(@(x)x.PlatformID, scenario.Platforms);
targets = scenario.Platforms(~ismember(platIDs,receiverIDs(:)));

if nargin < 3
    measNoise = 0;
end

if nargin < 4
    Pd = 1;
end

if nargin < 5
    nFalse = 0;
end

% False alarms per pair
if isscalar(nFalse)
    nFalse = nFalse*ones(nReceivers,1);
end

% Define sampleDetection
measParams = struct('OriginPosition',zeros(3,1));
sampleDetection = objectDetection(scenario.SimulationTime,0,'MeasurementParameters',measParams);

params = computeGetGlobalParameters();
emissionSpeed = params.EmissionSpeed;
timeScale = params.TimeScale;
detections = cell(0,1);

for i = 1:nReceivers
    sampleDetection.SensorIndex = i;
    thisID = receiverIDs(i);
    receiverPose = pose(scenario.Platforms{platIDs == thisID},'true');
    % True detections
    toas = zeros(numel(targets),1);
    isDetected = rand(numel(targets),1) < Pd;
    for j = 1:numel(targets)
        targetPose = pose(targets{j},'true');
        r1 = norm(targetPose.Position - receiverPose.Position);
        trueTOA = r1/emissionSpeed + scenario.SimulationTime;
        toas(j) = trueTOA*timeScale + sqrt(measNoise)*randn;
    end
    toas = toas(isDetected);
    % False TOA detections
    toaFalse = max(0,scenario.SimulationTime - (1/scenario.UpdateRate)*rand(nFalse(i),1))*timeScale;
    toas = [toas;toaFalse]; %#ok<AGROW>
    thisTOADetections = repmat({sampleDetection},numel(toas),1);
    for j = 1:numel(thisTOADetections)
        thisTOADetections{j}.Time = scenario.SimulationTime + 0.05; % Reporting in seconds
        thisTOADetections{j}.Measurement = toas(j);
        thisTOADetections{j}.MeasurementNoise = measNoise;
        thisTOADetections{j}.MeasurementParameters(1).OriginPosition = receiverPose.Position(:);
    end
    detections = [detections;thisTOADetections]; %#ok<AGROW> 
end

end
