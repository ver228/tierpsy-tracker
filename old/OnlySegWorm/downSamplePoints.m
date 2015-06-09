function [dsPoints dsIndices dsLengths] = ...
    downSamplePoints(points, samples, varargin)
%DOWNSAMPLEPOINTS Downsample the points to fewer points using the chain
%   code length.
%
%   INDICES = DOWNSAMPLEPOINTS(POINTS, CHAINCODELENGTHS, SAMPLES)
%
%   Inputs:
%       points           - the points to downsample
%       samples          - the number of samples to take
%       chainCodeLengths - the chain-code length at each point;
%                          if empty, the array indices are used instead
%
%   Output:
%       dsPoints  - the interpolated points for the samples based on their
%                   chain-code-length spacing
%       dsIndices - the indices for the samples based on their
%                   chain-code-length spacing
%       dsLengths - the chain-code lengths for the samples
%
% See also COMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are the points 2 dimensional?
if ndims(points) ~=2 || (size(points, 1) ~= 2 && size(points, 2) ~= 2)
    error('downSamplePoints:PointsNot2D', ...
        'The matrix of points must be 2 dimensional');
end

% Orient the points as a N-by-2 matrix.
isTransposed = false;
if size(points, 2) ~= 2
    points = points';
    isTransposed = true;
end

% Are there chain-code lengths?
if length(varargin) == 1
    chainCodeLengths = varargin{1};
else
    chainCodeLengths = 1:size(points, 1);
end

% Are we sampling enough points?
if samples < 1
    error('downSamplePoints:TooFewSamples', ...
        'The number of sampling points must be at least 1.');
end

% Are we sampling too many points?
if size(points, 1) < samples
    error('downSamplePoints:TooManySamples', ...
        ['The number of sampling points must be less than the number ' ...
         'of point pairs.']);
end

% Downsample the points to the middle.
if samples == 1
    dsLengths = (chainCodeLengths(end) + chainCodeLengths(1)) / 2;
    [dsPoints dsIndices] = chainCodeLengthInterp(points, dsLengths, ...
        chainCodeLengths);
    
% Downsample the points to the ends.
elseif samples == 2
    dsLengths = [chainCodeLengths(1); chainCodeLengths(end)];
    dsPoints = [points(1,:); points(end,:)];
    dsIndices = [1; size(points, 1)];
    
% Downsample the points to the requested number of samples.
% Note: we offset then scale the chain-code lengths so that they lie on the
% interval spanning 0 to 1. Then we divide this interval into the requested
% number of samples where 0 and 1 are, respectively, the first and last
% samples. Finally, we de-scale and de-offset the interval to obtain the
% chain-code length at each sample.
else
    
    % Compute the fractions for everything but the first and last sample.
    range = (0:(samples - 1))';
    fractions = range / (samples - 1);
    dsLengths = chainCodeLengths(1) + ...
        fractions * (chainCodeLengths(end) - chainCodeLengths(1));
    
    % Downsample the points.
    dsPoints(samples,:) = points(end,:);
    dsPoints(1,:) = points(1,:);
    dsIndices(samples,:) = size(points, 1);
    dsIndices(1,:) = 1;
    [dsPoints(range(3:end),:), dsIndices(range(3:end))] = ...
        chainCodeLengthInterp(points, dsLengths(2:end-1), chainCodeLengths);
end

% Transpose the points.
if isTransposed
    dsPoints = dsPoints';
    dsIndices = dsIndices';
    dsLengths = dsLengths';
end
end
