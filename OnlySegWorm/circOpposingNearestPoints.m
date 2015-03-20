function pointsI = circOpposingNearestPoints(pointsI, x, startI, endI, ...
    searchLength, varargin)
%CIRCOPPOSINGNEARESTPOINTS Find the nearest equivalent point indices on the
%   opposing side (within a search window) of a circular vector.
%
%   POINTSI = CIRCOPPOSINGNEARESTPOINTS(POINTSI, X, STARTI, ENDI,
%                                       SEARCHLENGTH)
%
%   POINTSI = CIRCOPPOSINGNERAESTPOINTS(POINTSI, X, STARTI, ENDI,
%                                       SEARCHLENGTH, CHAINCODELENGTHS)
%
%   Inputs:
%       pointsI          - the point indices to find on the opposing side
%       x                - the circularly connected vector on which the
%                          points lie
%       startI           - the index in the vector where the split, between
%                          opposing sides, starts
%       endI             - the index in the vector where the split, between
%                          opposing sides, ends
%       searchLength     - the search length, on either side of a directly
%                          opposing point, to search for the nearest point
%       chainCodeLengths - the chain-code length at each point;
%                          if empty, the array indices are used instead
%
%   Output:
%       pointsI - the equivalent point indices on the opposing side
%
% See also CIRCOPPOSINGPOINTS, CIRCNEARESTPOINTS, CIRCCOMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Re-order the start and end to make life simple. 
if startI > endI
    tmp = startI;
    startI = endI;
    endI = tmp;
end

% The points are degenerate.
if endI - startI < 2 || startI + size(x,1) - endI < 2
    pointsI = [];
    return;
end

% Are there chain-code lengths?
if length(varargin) == 1
    chainCodeLengths = varargin{1};
    
% Use the array indices for length.
else
    chainCodeLengths = 1:size(x,1);
end

% Compute the opposing points.
oPointsI = circOpposingPoints(pointsI, startI, endI, size(x,1), ...
    chainCodeLengths);

% Separate the points onto sides.
% Note: ignore start and end points, they stay the same.
% Side1 always goes from start to end in positive, index increments.
% Side2 always goes from start to end in negative, index increments.
side12 = oPointsI ~= startI & oPointsI ~= endI;
oPointsI(~side12) = [];
side1 = oPointsI > startI & oPointsI < endI;
side2 = oPointsI < startI | oPointsI > endI;

% Compute the start indices.
% Note: we checked for degeneracy; therefore, only one index can wrap.
is2Wrap = false;
start1 = startI + 1;
start2 = startI - 1;
if start2 < 1
    start2 = start2 + size(x,1);
    is2Wrap = true;
end

% Compute the end indices.
end1 = endI - 1;
end2 = endI + 1;
if end2 > size(x,1)
    end2 = end2 - size(x,1);
    is2Wrap = true;
end

% Compute the minimum search points on side 2 (for the search intervals
% opposite side 1).
minOPointsI(side1) = chainCodeLengths(oPointsI(side1)) - searchLength;
wrap = false(size(side1));
wrap(side1) = minOPointsI(side1) < chainCodeLengths(1);
minOPointsI(wrap) = minOPointsI(wrap) + chainCodeLengths(end);
wrap = false(size(side1));
wrap(side1) = minOPointsI(side1) < chainCodeLengths(start1) | ...
    minOPointsI(side1) > chainCodeLengths(end1);
minOPointsI(wrap) = start1;
notWrap = side1 & ~wrap;
minOPointsI(notWrap) = chainCodeLength2Index(minOPointsI(notWrap), ...
    chainCodeLengths);

% Compute the maximum search points on side 2 (for the search intervals
% opposite side 1).
maxOPointsI(side1) = chainCodeLengths(oPointsI(side1)) + searchLength;
wrap = false(size(side1));
wrap(side1) = maxOPointsI(side1) > chainCodeLengths(end);
maxOPointsI(wrap) = maxOPointsI(wrap) - chainCodeLengths(end);
wrap = false(size(side1));
wrap(side1) = maxOPointsI(side1) < chainCodeLengths(start1) | ...
    maxOPointsI(side1) > chainCodeLengths(end1);
maxOPointsI(wrap) = end1;
notWrap = side1 & ~wrap;
maxOPointsI(notWrap) = chainCodeLength2Index(maxOPointsI(notWrap), ...
    chainCodeLengths);

% Compute the minimum search points on side 1 (for the search intervals
% opposite side 2).
minOPointsI(side2) = chainCodeLengths(oPointsI(side2)) - searchLength;
wrap = false(size(side2));
wrap(side2) = minOPointsI(side2) < chainCodeLengths(1);
minOPointsI(wrap) = minOPointsI(wrap) + chainCodeLengths(end);
wrap = false(size(side2));
if is2Wrap
    wrap(side2) = minOPointsI(side2) > chainCodeLengths(start2) | ...
        minOPointsI(side2) < chainCodeLengths(end2);
else
    wrap(side2) = minOPointsI(side2) > chainCodeLengths(start2) & ...
        minOPointsI(side2) < chainCodeLengths(end2);
end
minOPointsI(wrap) = end2;
notWrap = side2 & ~wrap;
minOPointsI(notWrap) = chainCodeLength2Index(minOPointsI(notWrap), ...
    chainCodeLengths);

% Compute the maximum search points on side 1 (for the search intervals
% opposite side 2).
maxOPointsI(side2) = chainCodeLengths(oPointsI(side2)) + searchLength;
wrap = false(size(side2));
wrap(side2) = maxOPointsI(side2) > chainCodeLengths(end);
maxOPointsI(wrap) = maxOPointsI(wrap) - chainCodeLengths(end);
wrap = false(size(side2));
if is2Wrap
    wrap(side2) = maxOPointsI(side2) > chainCodeLengths(start2) | ...
        maxOPointsI(side2) < chainCodeLengths(end2);
else
    wrap(side2) = maxOPointsI(side2) > chainCodeLengths(start2) & ...
        maxOPointsI(side2) < chainCodeLengths(end2);
end
maxOPointsI(wrap) = start2;
notWrap = side2 & ~wrap;
maxOPointsI(notWrap) = chainCodeLength2Index(maxOPointsI(notWrap), ...
    chainCodeLengths);

% Search for the nearest points.
pointsI(side12) = circNearestPoints(x(pointsI(side12),:), minOPointsI, ...
    maxOPointsI, x);
end
