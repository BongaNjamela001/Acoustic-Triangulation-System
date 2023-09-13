function tdoaDetections = computeFormulateTDOAs(toaDetections, assignments)

% This function formulates TDOAs from TOA detections given known
% assignments. assignments is a N-by-P matrix, where each row defines the
% assignment between TOA detections. This assignment input can be
% calculated by the staticDetectionFuser as its second output. 

% Copyright 2021 The MathWorks, Inc. 

tdoaDetections = cell(0,1);

% Loop over all assignments
for i = 1:size(assignments,1)
    % Collect TOA detections in this assignment
    thisAssignment = assignments(i,:);
    thisAssignment = thisAssignment(thisAssignment > 0);
    thisDetections = toaDetections(thisAssignment);

    % Convert TOA to TDOA detections given they originate from the same
    % object
    thisTDOADetections = computeTOA2TDOADetections(thisDetections);

    % Fill the buffer
    tdoaDetections = [tdoaDetections;thisTDOADetections]; %#ok<AGROW> 
end

end
