function [contour] = bwClockTrace(img, seed, isClockwise)
%BWCLOCKTRACE Trace a contour (counter) clockwise.
%
%   [CONTOUR] = BWCLOCKTRACE(IMG, SEED, ISCLOCKWISE, <STARTDIR>)
%
%   Inputs:
%       img           - the binary image containing the connected component
%                       whose contour we are tracing
%       seed          - an [x y] seed pixel that lies on the contour
%       isClockwise   - true to trace clockwise and false to trace counter
%                       clockwise
%
%   Output:
%       contour - the directionally-traced continuous contour
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Setup a clock to check the 8-pixel neighborhood.
pixelClock = ...
{ -1  0 'N';
  -1  1 'NE';
   0  1 'E';
   1  1 'SE';
   1  0 'S';
   1 -1 'SW';
   0 -1 'W';
  -1 -1 'NW'};
%  y  x direction

% Does the seed lie on a boundary?
x = seed(1);
y = seed(2);
[maxY, maxX] = size(img);

% Cannot go west.
if x == 1
    
    % Go north.
    if isClockwise && y > 1
        j = 1;
            
    % Go south.
    elseif ~isClockwise && y < maxY
        j = 5;
        
    % Go east.
    else
        j = 3;
    end
    
% Cannot go east.
elseif x == maxX
    
    % Go south.
    if isClockwise && y < maxY
        j = 5;
            
    % Go north.
    elseif ~isClockwise && y > 1
        j = 1;
        
    % Go west.
    else
        j = 7;
    end
    
% Cannot go north.
elseif y == 1
    
    % Go east.
    if isClockwise && x < maxX
        j = 3;
            
    % Go west.
    elseif ~isClockwise && x > 1
        j = 7;
        
    % Go south.
    else
        j = 5;
    end
    
% Cannot go south.
elseif y == maxY
    
    % Go west.
    if isClockwise && x > 1
        j = 7;
            
    % Go east.
    elseif ~isClockwise && x < maxX
        j = 3;
        
    % Go north.
    else
        j = 1;
    end
    
% Find the next (counter) clockwise, contour point.
else
    % Find a background pixel.
    i = 1;
    clockLength = size(pixelClock, 1);
    while i <= clockLength && img(y + pixelClock{i,1}, x + pixelClock{i,2}) == 1
        i = i + 1;
    end
    
    % Does the seed lie on the contour?
    if i > clockLength
        error('bwClockTrace:NotContourPoint', 'The seed does not lie on the contour');
    end
    
    % Find the next clockwise, contour point.
    if isClockwise
        j = i + 1;
        if j > clockLength
            j = 1;
        end
        while j ~= i && img(y + pixelClock{j,1}, x + pixelClock{j,2}) == 0
            j = j + 1;
            if j > clockLength
                j = 1;
            end
        end
        
    % Find the next counter-clockwise, contour point.
    else
        j = i - 1;
        if j < 1
            j = clockLength;
        end
        while j ~= i && img(y + pixelClock{j,1}, x + pixelClock{j,2}) == 0
            j = j - 1;
            if j < 1
                j = clockLength;
            end
        end
    end
end

% Trace the contour.
if isClockwise
    contour = bwtraceboundary(img, ind2sub(size(img), [y x]), ...
        pixelClock{j,3}, 8, inf, 'clockwise');
else
    contour = bwtraceboundary(img, ind2sub(size(img), [y x]), ...
        pixelClock{j,3}, 8, inf, 'counterclockwise');
end
end
