function nearI = circNearestPoints(points, minI, maxI, x)
%CIRCNEARESTPOINTS For each point, find the nearest corresponding point
%   within an interval of circularly-connected search points.
%
%   NEARI = CIRCNEARESTPOINTS(POINTS, MINI, MAXI, X)
%
%   Inputs:
%       points - the point coordinates from which the distance is measured
%       minI   - the minimum indices of the intervals
%       maxI   - the maximum indices of the intervals
%       x      - the circularly-connected, point coordinates on which the
%                search intervals lie
%
%   Output:
%       nearI - the indices of the nearest points
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are the points 2 dimensional?
if ndims(points) ~=2 || (size(points, 1) ~= 2 && size(points, 2) ~= 2)
    error('circNearestPoints:PointsNot2D', ...
        'The matrix of points must be 2 dimensional');
end

% Are the search points 2 dimensional?
if length(size(x)) ~=2 || (size(x, 2) ~= 2 && size(x, 2) ~= 2)
    error('circNearestPoints:XNot2D', ...
        'The circularly-connected search points must be 2 dimensional');
end

% Orient the points as a N-by-2 matrix.
isTransposed = false;
if size(points, 2) ~= 2
    points = points';
    isTransposed = true;
end

% Orient the search points as a N-by-2 matrix.
if size(x, 2) ~= 2
    x = x';
end

% Pre-allocate memory.
nearI(1:size(points, 1)) = NaN;

% Search for the nearest points.
for i = 1:size(points, 1)
    
    % The interval is continuous.
    if minI(i) <= maxI(i)
        [~, nearI(i)] = min((points(i,1) - x(minI(i):maxI(i),1)).^ 2 + ...
            (points(i,2) - x(minI(i):maxI(i),2)) .^ 2);
        nearI(i) = nearI(i) + minI(i) - 1;
        
    % The interval wraps.
    else
        [mag1, nearI1] = min((points(i,1) - x(minI(i):end,1)) .^ 2 + ...
            (points(i,2) - x(minI(i):end,2)) .^ 2);
        [mag2, nearI2] = min((points(i,1) - x(1:maxI(i),1)) .^ 2 + ...
            (points(i,2) - x(1:maxI(i),2)) .^ 2);
        
        % Which point is nearest?
        if mag1 <= mag2
            nearI(i) = nearI1 + minI(i) - 1;
        else
            nearI(i) = nearI2;
        end
    end
end

% Transpose the point indices.
if isTransposed
    nearI = nearI';
end
end
