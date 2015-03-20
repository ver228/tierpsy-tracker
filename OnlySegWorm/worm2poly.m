function [polygon lcBounds rcBounds sBounds] = worm2poly(startSI, endSI, ...
    skeleton, headCI, tailCI, contour, isSplit, varargin)
%WORM2POLY Compute the polygon outline for a worm segment.
%
%   POLYGON = WORM2POLY(STARTSI, ENDSI, SKELETON, HEADCI, TAILCI, ...
%       CONTOUR)
%
%   POLYGON = WORM2POLY(STARTSI, ENDSI, SKELETON, HEADCI, TAILCI, ...
%       CONTOUR, SCCLENGTHS)
%
%   POLYGON = WORM2POLY(STARTSI, ENDSI, SKELETON, HEADCI, TAILCI, ...
%       CONTOUR, SCCLENGTHS, CCCLENGTHS)
%
%   Inputs:
%       startSI     - the skeleton index for the start of the worm segment
%       ensSI       - the skeleton index for the end of the worm segment
%       skeleton    - the worm's skeleton
%       headCI      - the contour index for the worm's head
%       tailCI      - the contour index for the worm's tail
%       contour     - the worm's contour
%       isSplit     - if true, split the worm segment, at the skeleton,
%                     into 2 halves (the 2 halves are returned as a single
%                     polygon of 2 cells)
%       sCCLengths  - the skeleton's chain-code length at each point;
%                     if empty, the array indices are used instead
%       cCCLengths  - the contour's chain-code length at each point;
%                     if empty, the array indices are used instead
%
%   Output:
%       polygon  - the polygon outline for the worm segment;
%                  or, if isSplit set to true, 2 cells where:
%                  polygon{1} = the side clockwise from the head, and
%                  polygon{2} = the side counter clockwise from the head
%       lcBounds - the left-side (counter clockwise from the head) contour
%                  bounds (the start and end indices of the segment)
%       rcBounds - the right-side (clockwise from the head) contour
%                  bounds (the start and end indices of the segment)
%       sBounds  - skeleton bounds (the start and end indices of the segment)
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are there chain-code lengths?
sLength = size(skeleton, 1);
cLength = size(contour, 1);
switch length(varargin)
    
    % Use chain-code lengths for the skeleton.
    % Use array-index length for the contour.
    case 1
        sCCLengths = varargin{1};
        cCCLengths = 1:cLength;
        
    % Use chain-code lengths for the skeleton and contour.
    case 2
        sCCLengths = varargin{1};
        cCCLengths = varargin{2};

    % Use array-index lengths for the skeleton and contour.
    otherwise
        sCCLengths = 1:sLength;
        cCCLengths = 1:cLength;
end

% Pre-compute values.
endSCCLengths = sCCLengths(end);
startCCCLengths = cCCLengths(1);
endCCCLengths = cCCLengths(end);
headCCLCI = cCCLengths(headCI);
tailCCLCI = cCCLengths(tailCI);

% Go from start to end in positive, index increments.
if startSI > endSI
    tmp = startSI;
    startSI = endSI;
    endSI = tmp;
end

% Compute the edge size to use in searching for opposing contour points.
% We use 1/4 of a contour side to be safe.
% Note: worm curvature can significantly distort the length of a contour
% side and, consequently, the locations of identical spots on opposing
% sides of the contour. Therefore, in addition to using scaled locations,
% we also use a large search window to ensure we correctly identify
% opposing contour locations.
searchEdgeSize = endCCCLengths / 8;

% Compute the polygon of the worm chunk.
% Side1 always goes from start to end in positive, index increments.
% Side2 always goes from start to end in negative, index increments.
if headCI < tailCI % side 1 is continuous
    
    % Compute the boundaries for each side.
    preHeadCI = headCI - 1;
    if preHeadCI < 1
        preHeadCI = cLength;
    end
    postHeadCI = headCI + 1;
    preTailCI = tailCI - 1;
    postTailCI = tailCI + 1;
    if postTailCI > cLength
        postTailCI = 1;
    end
    
    % Compute each side's size.
    size1 = tailCCLCI - headCCLCI;
    size2 = headCCLCI + endCCCLengths - tailCCLCI;
    
    % Scale the starting skeleton point to corresponding contour points.
    nStartSI = sCCLengths(startSI) / endSCCLengths;
    sCCL1 = headCCLCI + nStartSI * size1;
    eCCL2 = headCCLCI - nStartSI * size2;
    if eCCL2 < startCCCLengths
        eCCL2 = eCCL2 + endCCCLengths;
    end
    
    % Find the starting contour point for side 1.
    minSCCL1 = sCCL1 - searchEdgeSize;
    if minSCCL1 <= headCCLCI
        minS1 = postHeadCI;
    else
        minS1 = chainCodeLength2Index(minSCCL1, cCCLengths);
    end
    maxSCCL1 = sCCL1 + searchEdgeSize;
    if maxSCCL1 >= tailCCLCI
        maxS1 = preTailCI;
    else
        maxS1 = chainCodeLength2Index(maxSCCL1, cCCLengths);
    end
    startCI1 = circNearestPoints(skeleton(startSI,:), minS1, maxS1, ...
        contour);
    
    % Find the ending contour point for side 2.
    minECCL2 = eCCL2 - searchEdgeSize;
    if minECCL2 < startCCCLengths
        minECCL2 = minECCL2 + endCCCLengths;
    end
    if minECCL2 <= tailCCLCI && minECCL2 >= headCCLCI
        minE2 = postTailCI;
    else
        minE2 = chainCodeLength2Index(minECCL2, cCCLengths);
    end
    maxECCL2 = eCCL2 + searchEdgeSize;
    if maxECCL2 > endCCCLengths
        maxECCL2 = maxECCL2 - endCCCLengths;
    end
    if maxECCL2 >= headCCLCI && maxECCL2 <= tailCCLCI
        maxE2 = preHeadCI;
    else
        maxE2 = chainCodeLength2Index(maxECCL2, cCCLengths);
    end
    endCI2 = circNearestPoints(skeleton(startSI,:), minE2, maxE2, ...
        contour);
    
    % Scale the ending skeleton point to corresponding contour points.
    nEndSI = (endSCCLengths - sCCLengths(endSI)) / endSCCLengths;
    eCCL1 = tailCCLCI - nEndSI * size1;
    sCCL2 = tailCCLCI + nEndSI * size2;
    if sCCL2 > endCCCLengths
        sCCL2 = sCCL2 - endCCCLengths;
    end
    
    % Find the ending contour point for side 1.
    minECCL1 = eCCL1 - searchEdgeSize;
    if minECCL1 <= headCCLCI
        minE1 = postHeadCI;
    else
        minE1 = chainCodeLength2Index(minECCL1, cCCLengths);
    end
    maxECCL1 = eCCL1 + searchEdgeSize;
    if maxECCL1 >= tailCCLCI
        maxE1 = preTailCI;
    else
        maxE1 = chainCodeLength2Index(maxECCL1, cCCLengths);
    end
    endCI1 = circNearestPoints(skeleton(endSI,:), minE1, maxE1, contour);
    
    % Find the starting contour point for side 2.
    minSCCL2 = sCCL2 - searchEdgeSize;
    if minSCCL2 < startCCCLengths
        minSCCL2 = minSCCL2 + endCCCLengths;
    end
    if minSCCL2 <= tailCCLCI && minSCCL2 >= headCCLCI
        minS2 = postTailCI;
    else
        minS2 = chainCodeLength2Index(minSCCL2, cCCLengths);
    end
    maxSCCL2 = sCCL2 + searchEdgeSize;
    if maxSCCL2 > endCCCLengths
        maxSCCL2 = maxSCCL2 - endCCCLengths;
    end
    if maxSCCL2 >= headCCLCI && maxSCCL2 <= tailCCLCI
        maxS2 = preHeadCI;
    else
        maxS2 = chainCodeLength2Index(maxSCCL2, cCCLengths);
    end
    startCI2 = circNearestPoints(skeleton(endSI,:), minS2, maxS2, contour);
    
    % Initialize the start/end contour/skeleton points.
    startSP = skeleton(startSI,:);
    endSP = skeleton(endSI,:);
    startCP1 = contour(startCI1,:);
    endCP1 = contour(endCI1,:);
    startCP2 = contour(startCI2,:);
    endCP2 = contour(endCI2,:);
    
    % Compute the line from the end of contour side 2
    % to the start of the skeleton. 
    pointsEC2toSS = max(abs(endCP2 - startSP)) + 1;
    lineEC2toSS = round([linspace(endCP2(1), startSP(1), pointsEC2toSS); ...
        linspace(endCP2(2), startSP(2), pointsEC2toSS)]');

    % Compute the line from the start of the skeleton
    % to the start of contour side 1. 
    pointsSStoSC1 = max(abs(startSP - startCP1)) + 1;
    lineSStoSC1 = round([linspace(startSP(1), startCP1(1), pointsSStoSC1); ...
        linspace(startSP(2), startCP1(2), pointsSStoSC1)]');
    
    % Compute the line from the end of contour side 1
    % to the end of the skeleton. 
    pointsEC1toES = max(abs(endCP1 - endSP)) + 1;
    lineEC1toES = round([linspace(endCP1(1), endSP(1), pointsEC1toES); ...
        linspace(endCP1(2), endSP(2), pointsEC1toES)]');

    % Compute the line from the end of the skeleton
    % to the start of contour side 2. 
    pointsEStoSC2 = max(abs(endSP - startCP2)) + 1;
    lineEStoSC2 = round([linspace(endSP(1), startCP2(1), pointsEStoSC2); ...
        linspace(endSP(2), startCP2(2), pointsEStoSC2)]');
    
    % Construct the split polygon.
    if isSplit
%         % Construct the side clockwise from the head.
%         % Note: this side is continuous.
%         polygon{1} = [contour(startCI1:endCI1,:); ...
%             flipud(skeleton(startSI:endSI,:))];
%         
%         % Construct the side counter clockwise from the head.
%         if startCI2 < endCI2
%             polygon{2} = [contour(startCI2:endCI2,:); ...
%                 skeleton(startSI:endSI,:)];
%         else
%             polygon{2} = [contour(startCI2:end,:); ...
%                 contour(1:endCI2,:); skeleton(startSI:endSI,:)];
%         end
        
        % Construct the side clockwise from the head.
        % Note: this side is continuous.
        polygon{1} = [lineSStoSC1; contour(startCI1:endCI1,:); ...
            lineEC1toES; flipud(skeleton(startSI:endSI,:))];
        
        % Construct the side counter clockwise from the head.
        if startCI2 < endCI2
            polygon{2} = [lineEStoSC2; contour(startCI2:endCI2,:); ...
                lineEC2toSS; skeleton(startSI:endSI,:)];
        else
            polygon{2} = [lineEStoSC2; contour(startCI2:end,:); ...
                contour(1:endCI2,:); lineEC2toSS; skeleton(startSI:endSI,:)];
        end
        
    % Construct the polygon.
    % Note: side 1 is continuous.
    else
        if startCI2 < endCI2
%             polygon = [skeleton(startSI,:); contour(startCI1:endCI1,:); ...
%                 skeleton(endSI,:); contour(startCI2:endCI2,:)];

            polygon = [lineEC2toSS; lineSStoSC1; contour(startCI1:endCI1,:); ...
                lineEC1toES; lineEStoSC2; contour(startCI2:endCI2,:)];
        else
%             polygon = [skeleton(startSI,:); contour(startCI1:endCI1,:); ...
%                 skeleton(endSI,:); contour(startCI2:end,:); contour(1:endCI2,:)];

            polygon = [lineEC2toSS; lineSStoSC1; contour(startCI1:endCI1,:); ...
                lineEC1toES; lineEStoSC2; contour(startCI2:end,:); ...
                contour(1:endCI2,:)];
        end
    end
    
else % side 1 wraps
    
    % Compute the boundaries for each side.
    preHeadCI = headCI - 1;
    postHeadCI = headCI + 1;
    if postHeadCI > cLength
        postHeadCI = 1;
    end
    preTailCI = tailCI - 1;
    if preTailCI < 1
        preTailCI = cLength;
    end
    postTailCI = tailCI + 1;
    
    % Compute each side's size.
    size2 = headCCLCI - tailCCLCI;
    size1 = tailCCLCI + endCCCLengths - headCCLCI;
    
    % Scale the starting skeleton point to corresponding contour points.
    nStartSI = sCCLengths(startSI) / endSCCLengths;
    sCCL1 = headCCLCI + nStartSI * size1;
    if sCCL1 > endCCCLengths
        sCCL1 = sCCL1 - endCCCLengths;
    end
    eCCL2 = headCCLCI - nStartSI * size2;
    
    % Find the starting contour point for side 1.
    minSCCL1 = sCCL1 - searchEdgeSize;
    if minSCCL1 < startCCCLengths
        minSCCL1 = minSCCL1 + endCCCLengths;
    end
    if minSCCL1 <= headCCLCI && minSCCL1 >= tailCCLCI
        minS1 = postHeadCI;
    else
        minS1 = chainCodeLength2Index(minSCCL1, cCCLengths);
    end
    maxSCCL1 = sCCL1 + searchEdgeSize;
    if maxSCCL1 > endCCCLengths
        maxSCCL1 = maxSCCL1 - endCCCLengths;
    end
    if maxSCCL1 >= tailCCLCI && maxSCCL1 <= headCCLCI
        maxS1 = preTailCI;
    else
        maxS1 = chainCodeLength2Index(maxSCCL1, cCCLengths);
    end
    startCI1 = circNearestPoints(skeleton(startSI,:), minS1, maxS1, ...
        contour);
    
    % Find the ending contour point for side 2.
    minECCL2 = eCCL2 - searchEdgeSize;
    if minECCL2 <= tailCCLCI
        minE2 = postTailCI;
    else
        minE2 = chainCodeLength2Index(minECCL2, cCCLengths);
    end
    maxECCL2 = eCCL2 + searchEdgeSize;
    if maxECCL2 >= headCCLCI
        maxE2 = preHeadCI;
    else
        maxE2 = chainCodeLength2Index(maxECCL2, cCCLengths);
    end
    endCI2 = circNearestPoints(skeleton(startSI,:), minE2, maxE2, contour);
    
    % Scale the ending skeleton point to corresponding contour points.
    nEndSI = (endSCCLengths - sCCLengths(endSI)) / endSCCLengths;
    eCCL1 = tailCCLCI - nEndSI * size1;
    if eCCL1 < startCCCLengths
        eCCL1 = eCCL1 + endCCCLengths;
    end
    sCCL2 = tailCCLCI + nEndSI * size2;
    
    % Find the ending contour point for side 1.
    minECCL1 = eCCL1 - searchEdgeSize;
    if minECCL1 < startCCCLengths
        minECCL1 = minECCL1 + endCCCLengths;
    end
    if minECCL1 <= headCCLCI && minECCL1 >= tailCCLCI
        minE1 = postHeadCI;
    else
        minE1 = chainCodeLength2Index(minECCL1, cCCLengths);
    end
    maxECCL1 = eCCL1 + searchEdgeSize;
    if maxECCL1 > endCCCLengths
        maxECCL1 = maxECCL1 - endCCCLengths;
    end
    if maxECCL1 >= tailCCLCI && maxECCL1 <= headCCLCI
        maxE1 = preTailCI;
    else
        maxE1 = chainCodeLength2Index(maxECCL1, cCCLengths);
    end
    endCI1 = circNearestPoints(skeleton(endSI,:), minE1, maxE1, contour);
        
    % Find the starting contour point for side 2.
    minSCCL2 = sCCL2 - searchEdgeSize;
    if minSCCL2 <= tailCCLCI
        minS2 = postTailCI;
    else
        minS2 = chainCodeLength2Index(minSCCL2, cCCLengths);
    end
    maxSCCL2 = sCCL2 + searchEdgeSize;
    if maxSCCL2 >= headCCLCI
        maxS2 = preHeadCI;
    else
        maxS2 = chainCodeLength2Index(maxSCCL2, cCCLengths);
    end
    startCI2 = circNearestPoints(skeleton(endSI,:), minS2, maxS2, contour);
    
    % Initialize the start/end contour/skeleton points.
    startSP = skeleton(startSI,:);
    endSP = skeleton(endSI,:);
    startCP1 = contour(startCI1,:);
    endCP1 = contour(endCI1,:);
    startCP2 = contour(startCI2,:);
    endCP2 = contour(endCI2,:);
    
    % Compute the line from the end of contour side 2
    % to the start of the skeleton. 
    pointsEC2toSS = max(abs(endCP2 - startSP)) + 1;
    lineEC2toSS = round([linspace(endCP2(1), startSP(1), pointsEC2toSS); ...
        linspace(endCP2(2), startSP(2), pointsEC2toSS)]');

    % Compute the line from the start of the skeleton
    % to the start of contour side 1. 
    pointsSStoSC1 = max(abs(startSP - startCP1)) + 1;
    lineSStoSC1 = round([linspace(startSP(1), startCP1(1), pointsSStoSC1); ...
        linspace(startSP(2), startCP1(2), pointsSStoSC1)]');
    
    % Compute the line from the end of contour side 1
    % to the end of the skeleton. 
    pointsEC1toES = max(abs(endCP1 - endSP)) + 1;
    lineEC1toES = round([linspace(endCP1(1), endSP(1), pointsEC1toES); ...
        linspace(endCP1(2), endSP(2), pointsEC1toES)]');

    % Compute the line from the end of the skeleton
    % to the start of contour side 2. 
    pointsEStoSC2 = max(abs(endSP - startCP2)) + 1;
    lineEStoSC2 = round([linspace(endSP(1), startCP2(1), pointsEStoSC2); ...
        linspace(endSP(2), startCP2(2), pointsEStoSC2)]');
    
    % Construct the split polygon.
    if isSplit
%         % Construct the side clockwise from the head.
%         if startCI1 < endCI1
%             polygon{1} = [contour(startCI1:endCI1,:); ...
%                 flipud(skeleton(startSI:endSI,:))];
%         else
%             polygon{1} = [contour(startCI1:end,:); ...
%                 contour(1:endCI1,:); flipud(skeleton(startSI:endSI,:))];
%         end
%         
%         % Construct the side counter clockwise from the head.
%         % Note: this side is continuous.
%         polygon{2} = [contour(startCI2:endCI2,:); skeleton(startSI:endSI,:)];
        
        % Construct the side clockwise from the head.
        if startCI1 < endCI1
            polygon{1} = [lineSStoSC1; contour(startCI1:endCI1,:); ...
                lineEC1toES; flipud(skeleton(startSI:endSI,:))];
        else
            polygon{1} = [lineSStoSC1; contour(startCI1:end,:); ...
                contour(1:endCI1,:); lineEC1toES; flipud(skeleton(startSI:endSI,:))];
        end
        
        % Construct the side counter clockwise from the head.
        % Note: this side is continuous.
        polygon{2} = [lineEStoSC2; contour(startCI2:endCI2,:); ...
            lineEC2toSS; skeleton(startSI:endSI,:)];
        
    % Construct the polygon.
    % Note: side 2 is continuous.
    else
        if startCI1 < endCI1
%             polygon = [skeleton(startSI,:); contour(startCI1:endCI1,:); ...
%                 skeleton(endSI,:); contour(startCI2:endCI2,:)];
            
            polygon = [lineEC2toSS; lineSStoSC1; contour(startCI1:endCI1,:); ...
                lineEC1toES; lineEStoSC2; contour(startCI2:endCI2,:)];
        else
%             polygon = [skeleton(startSI,:); contour(startCI1:end,:); ...
%                 contour(1:endCI1,:); skeleton(endSI,:); contour(startCI2:endCI2,:)];

            polygon = [lineEC2toSS; lineSStoSC1; contour(startCI1:end,:); ...
                contour(1:endCI1,:); lineEC1toES; lineEStoSC2; ...
                contour(startCI2:endCI2,:)];
        end
    end
end

% Construct the bounds.
rcBounds = [startCI1; endCI1];
lcBounds = [startCI2; endCI2];
sBounds = [startSI; endSI];
end
