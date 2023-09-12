function parameters = computeGetGlobalParameters()

% This function defines the emissions speed and units for time used in the
% TDOA tracking example. Modify these values to use a different emission
% speed and time units. 

% Copyright 2021 The MathWorks, Inc. 

parameters.EmissionSpeed = 350;
% parameters.EmissionSpeed = 299792458.0;
parameters.TimeScale  = 50;
end
