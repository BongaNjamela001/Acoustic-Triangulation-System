function [tdoaDetections, isValid] = computeTOA2TDOADetections(toaDetections)

% This function converts multiple TOA detections from the same object to
% TDOA detections assuming 1 reference sensor. This allows it to directly
% use spherical intersection algorithm. 

% Copyright 2021 The MathWorks, Inc.
sampleTDOADetection = objectDetection(toaDetections{1}.Time,toaDetections{1}.Measurement,...
    'MeasurementParameters',repmat(toaDetections{1}.MeasurementParameters,2,1));

% Number of TOA detections
n = numel(toaDetections);

% At least 2 detections must be used for TDOA calculation
assert(n > 1, 'At least 2 TOA Detections required to form a TDOA');

% Allocate memory for TDOA detection
tdoaDetections = repmat({sampleTDOADetection},n-1,1);

% Position of the reference receiver (first TOA in the list)
referenceParams = toaDetections{1}.MeasurementParameters;
referencePos = referenceParams.OriginPosition;

% A pair of TOA may combine to form an invalid TDOA because its greater
% than the inter distance between receivers. 
isValid = false(1,n-1);

% Get emission speed and time scale
globalParams = computeGetGlobalParameters();
emissionSpeed = globalParams.EmissionSpeed;
timeScale = globalParams.TimeScale;

% Fill TDOA detections
for i = 2:numel(toaDetections)
    % Position of receiver
    receiverPos = toaDetections{i}.MeasurementParameters.OriginPosition;

    % TDOA = TOA - TOAreference
    tdoaDetections{i-1}.Measurement = toaDetections{i}.Measurement - toaDetections{1}.Measurement; % TDOA

    % Noise additions
    tdoaDetections{i-1}.MeasurementNoise = toaDetections{i}.MeasurementNoise + toaDetections{1}.MeasurementNoise;

    % Use SensorIndex of the non-reference receiver
    tdoaDetections{i-1}.SensorIndex = toaDetections{i}.SensorIndex;

    % Fill MeasurementParameters
    tdoaDetections{i-1}.MeasurementParameters(1) = toaDetections{i}.MeasurementParameters;
    tdoaDetections{i-1}.MeasurementParameters(2) = referenceParams;

    % Check if a TDOA is valid
    isValid(i-1) = norm(receiverPos - referencePos)/emissionSpeed*timeScale > abs(tdoaDetections{i-1}.Measurement);
end

end
