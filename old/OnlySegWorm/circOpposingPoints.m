function pointsI = circOpposingPoints(pointsI, startI, endI, vLength, ...
    varargin)
%CIRCOPPOSINGPOINTS Find the equivalent point indices on the opposing side
%   of a circular vector.
%
%   POINTSI = CIRCOPPOSINGPOINTS(POINTSI, STARTI, ENDI, VSIZE)
%
%   POINTSI = CIRCOPPOSINGPOINTS(POINTSI, STARTI, ENDI, VSIZE,
%                                CHAINCODELENGTHS)
%
%   Inputs:
%       pointsI          - the point indices to find on the opposing side
%       startI           - the index in the vector where the split, between
%                          opposing sides, starts
%       endI             - the index in the vector where the split, between
%                          opposing sides, ends
%       vLength          - the vector length
%       chainCodeLengths - the chain-code length at each point;
%                          if empty, the array indices are used instead
%
%   Output:
%       pointsI - the equivalent point indices on the opposing side
%
% See also CIRCCOMPUTECHAINCODELENGTHS
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
if endI - startI < 2 || startI + vLength - endI < 2
    pointsI = [];
    return;
end

% Are there chain-code lengths?
if length(varargin) == 1
    chainCodeLengths = varargin{1};
    
% Use the array indices for length.
else
    chainCodeLengths = 1:vLength;
end

% Separate the points onto sides.
% Note: ignore start and end points, they stay the same.
% Side1 always goes from start to end in positive, index increments.
% Side2 always goes from start to end in negative, index increments.
side1 = pointsI > startI & pointsI < endI;
side2 = pointsI < startI | pointsI > endI;
side12 = side1 | side2;

% Compute the size of side 1.
start1I = startI + 1;
end1I = endI - 1;
sSize1 = chainCodeLengths(end1I) - chainCodeLengths(start1I);

% Compute the size of side 2.
start2I = startI - 1;
if start2I < 1
    start2I = vLength;
end
end2I = endI + 1;
if end2I > vLength
    end2I = 1;
end
if start2I < end2I
    sSize2 = chainCodeLengths(start2I) + ...
        chainCodeLengths(vLength) - chainCodeLengths(end2I);
else % one of the ends wrapped
    sSize2 = chainCodeLengths(start2I) - chainCodeLengths(end2I);
end

% Compute the scale between sides.
scale1to2 = sSize2 / sSize1;
scale2to1 = sSize1 / sSize2;

% Find the distance of the side 1 points from the start, scale them for
% side 2, then find the equivalent point, at the scaled distance
% from the start, on side 2.
pointsI(side1) = chainCodeLengths(start2I) - ...
    (chainCodeLengths(pointsI(side1)) - chainCodeLengths(start1I)) ...
    * scale1to2;

% Find the distance of the side 2 points from the start, scale them for
% side 1, then find the equivalent point, at the scaled distance
% from the start, on side 1.
minPoints2 = pointsI(side2) <= start2I;
minSide2 = false(length(pointsI),1);
maxSide2 = false(length(pointsI),1);
minSide2(side2) = minPoints2;
maxSide2(side2) = ~minPoints2;
pointsI(minSide2) = chainCodeLengths(start1I) + ...
    (chainCodeLengths(start2I) - chainCodeLengths(pointsI(minSide2))) ...
    * scale2to1;
pointsI(maxSide2) = chainCodeLengths(start1I) + ...
    (chainCodeLengths(start2I) + chainCodeLengths(vLength) - ...
    chainCodeLengths(pointsI(maxSide2))) * scale2to1;

% Correct any wrapped points.
wrap(side12) = pointsI(side12) < 0;
pointsI(wrap) = pointsI(wrap) + chainCodeLengths(vLength);
wrap(side12) = pointsI(side12) > chainCodeLengths(vLength);
pointsI(wrap) = pointsI(wrap) - chainCodeLengths(vLength);

% Translate the chain-code lengths to indices.
pointsI(side12) = chainCodeLength2Index(pointsI(side12), chainCodeLengths);
end
