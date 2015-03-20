function center = circPolyCenter(s1, e1, side1, s2, e2, side2, varargin)
%CIRCPOLYCENTER Find the center of a polygon between two circularly
%   continuous segments of its sides.
%
%   CENTER = CIRCPOLYCENTER(S1, E1, SIDE1, S2, E2, SIDE2)
%
%   CENTER = CIRCPOLYCENTER(S1, E1, SIDE1, S2, E2, SIDE2,
%                           CHAINCODELENGTHS1, CHAINCODELENGTHS2)
%
%   Note: the segments are circularly continuous; therefore, if the start
%   index is greater then the end index, the segment wraps around.
%
%   Inputs:
%       s1                - the start index for the segment on side 1
%       e1                - the end index for the segment on side 1
%       side1             - side 1's pixels
%       s2                - the start index for the segment on side 2
%       e2                - the end index for the segment on side 2
%       side2             - side 2's pixels
%       chainCodeLengths1 - the chain-code length at each point for side 1;
%                           if empty, the array indices are used instead
%       chainCodeLengths2 - the chain-code length at each point for side 2;
%                           if empty, the array indices are used instead
%
%   Output:
%       center - the center pixels between both circularly continuous
%                segments of the polygon's sides
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are there chain-code lengths?
if length(varargin) == 2
    chainCodeLengths1 = varargin{1};
    chainCodeLengths2 = varargin{2};
else
    chainCodeLengths1 = [];
    chainCodeLengths2 = [];
end

% Compute the center for side 1 using the array indices.
if isempty(chainCodeLengths1)
    if s1 <= e1
        center1I = ceil((s1 + e1) / 2);
    else
        length1 = size(side1, 1);
        center1I = ceil((s1 + e1 + length1) / 2);
        if center1I > length1
            center1I = center1I - length1;
        end
    end
    
% Compute the center for side 1 using the chain code lengths.
else
    if s1 <= e1
        centerCCL1 = (chainCodeLengths1(s1) + chainCodeLengths1(e1)) / 2;
    else
        centerCCL1 = (chainCodeLengths1(s1) + chainCodeLengths1(e1) + ...
            chainCodeLengths1(end)) / 2;
        if centerCCL1 > chainCodeLengths1(end)
            centerCCL1 = centerCCL1 - chainCodeLengths1(end);
        end
    end
    center1I = chainCodeLength2Index(centerCCL1, chainCodeLengths1);
end

% Compute the center for side 2 using the array indices.
if isempty(chainCodeLengths2)
    if s2 <= e2
        center2I = ceil((s2 + e2) / 2);
    else
        length2 = size(side2, 1);
        center2I = ceil((s2 + e2 + length2) / 2);
        if center2I > length2
            center2I = center2I - length2;
        end
    end
    
% Compute the center for side 2 using the chain code lengths.
else
    if s2 <= e2
        centerCCL2 = (chainCodeLengths2(s2) + chainCodeLengths2(e2)) / 2;
    else
        centerCCL2 = (chainCodeLengths2(s2) + chainCodeLengths2(e2) + ...
            chainCodeLengths2(end)) / 2;
        if centerCCL2 > chainCodeLengths2(end)
            centerCCL2 = centerCCL2 - chainCodeLengths2(end);
        end
    end
    center2I = chainCodeLength2Index(centerCCL2, chainCodeLengths2);
end

% Compute the center between both sides.
center = round((side1(center1I,:) + side2(center2I,:)) / 2);
end
