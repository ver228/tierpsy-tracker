function [cContour] = cleanContour(contour)
%CLEANCONTOUR Clean an 8-connected, circularly-connected contour by
%removing any duplicate points and interpolating any missing points.
%
%   [ccontour] = cleancontour(contour)
%
%   Input:
%       contour - the 8-connected, circularly-connected contour to clean
%
%   Output:
%       cContour - the cleaned contour (no duplicates & no missing points)
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Construct the cleaned contour.
cContour = zeros(size(contour));
j = 1;
for i = 1:(length(contour) - 1)
    
    % Initialize the point differences.
    y1 = contour(i,1);
    y2 = contour(i + 1,1);
    x1 = contour(i,2);
    x2 = contour(i + 1,2);
    y = abs(y2 - y1);
    x = abs(x2 - x1);
    
    % Ignore duplicates.
    if y == 0 && x == 0
        continue;
    end
    
    % Add the point.
    if (y == 0 || y == 1) && (x == 0 || x == 1)
        cContour(j,:) = contour(i,:);
        j = j + 1;
        
    % Interpolate the missing points.
    else
        points = max(y, x);
        cContour(j:(j + points),1) = round(linspace(y1, y2, points + 1));
        cContour(j:(j + points),2) = round(linspace(x1, x2, points + 1));
        j = j + points;
    end
end

% Add the last point
if (cContour(1,1) ~= contour(end,1)) || ...
        (cContour(1,2) ~= contour(end,2))
    cContour(j,:) = contour(end,:);
    j = j + 1;
end
cContour(j:end,:) = [];
end
