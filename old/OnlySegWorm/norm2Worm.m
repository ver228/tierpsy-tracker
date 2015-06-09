function worm = norm2Worm(frame, vulvaContour, nonVulvaContour, ...
    skeleton, skeletonAngles, inOutTouch, skeletonLength, widths, ...
    headArea, tailArea, vulvaArea, nonVulvaArea, ...
    origin, pixel2MicronScale, rotation, worm)
%NORM2WORM Convert normalized worm information into a worm structure.
%
%   WORM = NORM2WORM(FRAME, VULVACONTOUR, NONVULVACONTOUR, SKELETON,
%                    SKELETONANGLES, INOUTTOUCH, SKELETONLENGTH, WIDTHS,
%                    HEADAREA, TAILAREA, VULVAAREA, NONVULVAAREA,
%                    ORIGIN, PIXEL2MICRONSCALE, ROTATION, WORM)
%
%   Inputs:
%
%       Note 1: all inputs must be oriented head to tail.
%       Note 2: all coordinates must be oriented as (x,y); in contrast, in
%       the worm structure, coordinates are oriented as (row,column) = (y,x).
%
%       frame            - the worm's video frame
%       vulvaContour     - the worm's downsampled vulval-side contour
%       nonVulvaContour  - the worm's downsampled non-vulval-side contour
%       skeleton         - the worm's downsampled skeleton
%       angles           - the worm's downsampled skeleton angles
%                          Note: positive skeleton angles bulge towards the
%                          vulva; in contrast, in the worm structure,
%                          positive skeleton angles bulge towards the side
%                          clockwise from the worm's head (unless the worm
%                          is flipped).
%       inOutTouch       - in coiled worms, for each skeleton sample:
%                          -1 = the contours are inside the coil
%                          -2 = the contours are outside the coil
%                          -3 = the contours are both inside and outside
%                               the coil (one half of the contour is inside
%                               the coil, the other half of the contour is
%                               outside it)
%                          1+ = the contour is touching itself to form the
%                               coil; the specific number represents the
%                               contraposed skeleton index being touched
%       skeletonLength    - the worm's skeleton chain-code pixel lengths
%       widths            - the worms' contour downsampled widths
%       headArea          - the worms' head area
%       tailArea          - the worms' tail area
%       vulvaArea         - the worms' vulval-side area
%                           (excluding the head and tail)
%       nonVulvaArea      - the worm's non-vulval-side area
%                           (excluding the head and tail)
%       origin            - the real-world micron origin (stage location)
%                           for the worm (see findStageMovements, locations)
%       pixel2MicronScale - the scale for converting pixels to microns
%                           (see readPixels2Microns, pixel2MicronScale)
%       rotation          - the rotation matrix for onscreen pixels
%                           (see readPixels2Microns, rotation)
%       worm              - the original, pre-normalized, worm information
%                           organized in a structure; if empty, the worm is
%                           re-assembled as best as possible.
%
%   Output:
%       worm - the worm information organized in a structure
%              This structure contains 8 sub-structures,
%              6 sub-sub-structures, and 4 sub-sub-sub-structures:
%
%              * Video *
%              video = {frame}
%
%              * Contour *
%              contour = {pixels, touchI, inI, outI, angles, headI, tailI,
%                         chainCodeLengths}
%
%              * Skeleton *
%              skeleton = {pixels, touchI, inI, outI, inOutI, angles,
%                          length, chainCodeLengths, widths}
%
%              Note: positive skeleton angles bulge towards the side
%              clockwise from the worm's head (unless the worm is flipped).
%
%              * Head *
%              head = {bounds, pixels, area,
%                      cdf (at [2.5% 25% 50% 75% 97.5%]), stdev}
%              head.bounds{contour.left (indices for [start end]),
%                          contour.right (indices for [start end]),
%                          skeleton indices for [start end]}
%
%              * Tail *
%              tail = {bounds, pixels, area,
%                      cdf (at [2.5% 25% 50% 75% 97.5%]), stdev}
%              tail.bounds{contour.left (indices for [start end]),
%                          contour.right (indices for [start end]),
%                          skeleton indices for [start end]}
%
%              * Left Side (Counter Clockwise from the Head) *
%              left = {bounds, pixels, area,
%                      cdf (at [2.5% 25% 50% 75% 97.5%]), stdev}
%              left.bounds{contour (indices for [start end]),
%                          skeleton (indices for [start end])}
%
%              * Right Side (Clockwise from the Head) *
%              right = {bounds, pixels, area,
%                       cdf (at [2.5% 25% 50% 75% 97.5%]), stdev}
%              right.bounds{contour (indices for [start end]),
%                           skeleton (indices for [start end])}
%
%              * Orientation *
%              orientation = {head, vulva}
%              orientation.head = {isFlipped,
%                                  confidence.head, confidence.tail}
%              orientation.vulva = {isClockwiseFromHead,
%                                  confidence.vulva, confidence.nonVulva}
%
% See also NORMWORMS, READPIXELS2MICRONS, WORM2STRUCT
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Convert the worm to a structure.
% Note: if the original, pre-normalized, worm is unavailable, we
% re-assemble the worm with the head and tail not flipped and the vulval
% side clockwise from the head.
if ~isempty(worm) && ~isstruct(worm)
    worm = cell2worm(worm);
end

% The worm is roughly divided into 24 segments of musculature (i.e., hinges
% that represent degrees of freedom) on each side. Therefore, 48 segments
% around a 2-D contour.
% Note: "In C. elegans the 95 rhomboid-shaped body wall muscle cells are
% arranged as staggered pairs in four longitudinal bundles located in four
% quadrants. Three of these bundles (DL, DR, VR) contain 24 cells each,
% whereas VL bundle contains 23 cells." - www.wormatlas.org
sWormSegs = 24;
cWormSegs = 2 * sWormSegs;

% Pre-compute values.
pixel2MicronArea = sum(pixel2MicronScale .^ 2) / 2;
pixel2MicronMagnitude = sqrt(pixel2MicronArea);

% Convert the contours back to pixel coordinates.
vulvaContour = round(fliplr(microns2Pixels(origin, vulvaContour, ...
    pixel2MicronScale, rotation)));
nonVulvaContour = round(fliplr(microns2Pixels(origin, nonVulvaContour, ...
    pixel2MicronScale, rotation)));

% Convert the skeleton back to pixel coordinates.
skeleton = round(fliplr(microns2Pixels(origin, skeleton, ...
    pixel2MicronScale, rotation)));

% Compute the orientation.
samples = size(skeleton,1);
if isempty(worm)
    isHeadFlipped = false;
    isVulvaClockwiseFromHead = true;
    hConfidence = 0;
    tConfidence = 0;
    vConfidence = 0;
    nvConfidence = 0;
    
    % Is the vulva clockwise from the head?
    if samples > 2
        
        % The line separating the head and tail is at y = dHT(1).
        % The distance of the contour sides from this line determines which
        % side is clockwise from the head.
        dHT = skeleton(1,:) - skeleton(end,:);
        if dHT(1) == 0
            if max(vulvaContour(2:(end - 1),1)) > ...
                    max(nonVulvaContour(2:(end - 1),1))
                if dHT(2) < 0
                    isVulvaClockwiseFromHead = false;
                end
            elseif dHT(2) > 0
                isVulvaClockwiseFromHead = false;
            end
            
        % The line separating the head and tail is at x = dHT(2).
        % The distance of the contour sides from this line determines which
        % side is clockwise from the head.
        elseif dHT(2) == 0
            if max(vulvaContour(2:(end - 1),2)) < ...
                    max(nonVulvaContour(2:(end - 1),2))
                if dHT(1) < 0
                    isVulvaClockwiseFromHead = false;
                end
            elseif dHT(1) > 0
                isVulvaClockwiseFromHead = false;
            end
        
        % The line separating the head and tail is at:
        % y = m * x + b
        % where m = y / x and b = y - m * x
        %
        % The perpendicular line is:
        % y = -x / m
        %
        % The distance of the contour sides from this line determines which
        % side is clockwise from the head. Therefore, for each contour
        % point, we need to determine its perpendicular distance from the
        % line separating the head and tail. The perpendicular line is at:
        % y = -x / m + b2
        % where b2 = y + x / m
        % 
        % For each contour point, its perpendicular matching point on the
        % line separating the head and tail lies at the intersection of
        % these two lines (the head-to-tail line and the perpendicular line
        % passing through the contour point). Therefore, solving both
        % equations for the perpendicularly matching x and y, we get:
        % 0 = m * x + x / m + b - b2
        % x * (m + 1 / m) = b2 - b
        % x = (b2 - b) / (m + 1 / m)
        % and, y =  m * x + b
        else
            
            % Compute the line from head to tail.
            m = dHT(1) / dHT(2);
            b = skeleton(1,1) - m * skeleton(1,2);
            
            % Compute the perpendicular lines for the contour points.
            % Note: we ignore the head and tail.
            bVulva = vulvaContour(2:(end - 1),1) + ...
                vulvaContour(2:(end - 1),2) / m;
            bNonVulva = nonVulvaContour(2:(end - 1),1) + ...
                nonVulvaContour(2:(end - 1),2) / m;
            
            % Compute the perpendicular points for the contour points.
            pVulva(:,2) = (bVulva - b) / (m + 1 / m);
            pVulva(:,1) = m * pVulva(:,2) + b;
            pNonVulva(:,2) = (bNonVulva - b) / (m + 1 / m);
            pNonVulva(:,1) = m * pNonVulva(:,2) + b;
            
            % Compute the perpendicular distances for the contour points
            % from the head-to-tail line.
            dVulva = sum((vulvaContour(2:(end - 1),:) - pVulva) .^ 2, 2);
            dNonVulva = sum((nonVulvaContour(2:(end - 1),:) - ...
                pNonVulva) .^ 2, 2);
            
            % Which contour side has the furthest point from the
            % head-to-tail line?
            dHTSign = sign(dHT);
            [mVulva mVulvaI] = max(dVulva);
            [mNonVulva mNonVulvaI] = max(dNonVulva);
            if mVulva > mNonVulva
                dPSign = sign(vulvaContour(mVulvaI + 1,:) - ...
                    pVulva(mVulvaI,:));
            else
                dPSign = sign(pNonVulva(mNonVulvaI,:) - ...
                    nonVulvaContour(mNonVulvaI + 1,:));
            end
            
            % The head-to-tail line runs northeast to southwest.
            if dHTSign(1) == dHTSign(2)
                if dHTSign(2) == dPSign(2)
                    isVulvaClockwiseFromHead = false;
                end
                
            % The head-to-tail line runs southeast to northwest.
            elseif dHTSign(1) == dPSign(1)
                isVulvaClockwiseFromHead = false;
            end
        end
    end
    
% Determine the orientation.
else % ~isempty(worm)
    isHeadFlipped = worm.orientation.head.isFlipped;
    isVulvaClockwiseFromHead = worm.orientation.vulva.isClockwiseFromHead;
    hConfidence = worm.orientation.head.confidence.head;
    tConfidence = worm.orientation.head.confidence.tail;
    vConfidence = worm.orientation.vulva.confidence.vulva;
    nvConfidence = worm.orientation.vulva.confidence.nonVulva;
end

% Interpolate the contours' total chain-code length.
if isempty(worm)
    
    % The samples are at the midpoints.
    if samples == 1
        vLength = sqrt(sum((vulvaContour - nonVulvaContour) .^ 2, 2));
        nvLength = vLength;
        
    % The samples are at the head and tail.
    elseif samples == 2
        vLength = sqrt(sum(diff(vulvaContour) .^ 2, 2));
        nvLength = vLength;
        
    % The samples divide the worm evenly between its head and tail.
    else
        vLength = sqrt(max(sum(diff(vulvaContour) .^ 2, 2))) * ...
            (samples - 1);
        nvLength = sqrt(max(sum(diff(nonVulvaContour) .^ 2, 2))) * ...
            (samples - 1);
    end
    
% Compute the contours' total chain-code length.
else % ~isempty(worm)
    
    % Compute each side of the contour's total chain-code length.
    % Side1 always goes from head to tail in positive, index increments.
    % Side2 always goes from head to tail in negative, index increments.
    if worm.contour.headI <= worm.contour.tailI
        side1Length = ...
            worm.contour.chainCodeLengths(worm.contour.tailI) - ...
            worm.contour.chainCodeLengths(worm.contour.headI);
        side2Length = ...
            worm.contour.chainCodeLengths(worm.contour.headI) - ...
            worm.contour.chainCodeLengths(1) + ...
            worm.contour.chainCodeLengths(end) - ...
            worm.contour.chainCodeLengths(worm.contour.tailI) + ...
            worm.contour.chainCodeLengths(1);
        
    else % worm.contour.headI > worm.contour.tailI
        side2Length = ...
            worm.contour.chainCodeLengths(worm.contour.headI) - ...
            worm.contour.chainCodeLengths(worm.contour.tailI);
        side1Length = ...
            worm.contour.chainCodeLengths(worm.contour.tailI) - ...
            worm.contour.chainCodeLengths(1) + ...
            worm.contour.chainCodeLengths(end) - ...
            worm.contour.chainCodeLengths(worm.contour.headI) + ...
            worm.contour.chainCodeLengths(1);
    end
    
    % Determine the vulval and non-vulval contours' total chain-code lengths.
    if isVulvaClockwiseFromHead == isHeadFlipped
        vLength = side2Length;
        nvLength = side1Length;
    else
        vLength = side1Length;
        nvLength = side2Length;
    end
end

% Compute the chain-code lengths.
if samples == 1 % The samples are at the midpoints.
    sCCLengths = skeletonLength;
    vCCLengths = vLength / 2;
    nvCCLengths = nvLength / 2;
elseif samples == 2  % The samples are the head and tail.
    sCCLengths = [0; skeletonLength];
    vCCLengths = [0; vLength];
    nvCCLengths = [0; nvLength];
else % The samples divide the worm evenly between its head and tail.
    sCCLengths = ((0:(samples - 1))' * skeletonLength) / (samples - 1);
    vCCLengths = ((0:(samples - 1))' * vLength) / (samples - 1);
    nvCCLengths = ((0:(samples - 1))' * nvLength) / (samples - 1);
end

% Compute the clockwise and anti-clockwise sides from the head.
if isVulvaClockwiseFromHead
    clockPixels = vulvaContour;
    clockLengths = vCCLengths;
    antiPixels = flipud(nonVulvaContour);
    antiLengths = nvCCLengths;
else
    clockPixels = nonVulvaContour;
    clockLengths = nvCCLengths;
    antiPixels = flipud(vulvaContour);
    antiLengths = vCCLengths;
end

% Split the contour into sides.
% Side1 always goes from head to tail in positive, index increments.
% Side2 always goes from head to tail in negative, index increments.
if isHeadFlipped
    pixels1 = antiPixels;
    lengths1 = antiLengths;
    length1 = antiLengths(end);
    pixels2 = clockPixels;
    lengths2 = clockLengths;
    length2 = clockLengths(end);
else
    pixels1 = clockPixels;
    lengths1 = clockLengths;
    length1 = clockLengths(end);
    pixels2 = antiPixels;
    lengths2 = antiLengths;
    length2 = antiLengths(end);
end

% Compute the contour's pixels, chain-code lengths, and angles.
if isempty(worm) || worm.contour.headI <= worm.contour.tailI
    
    % The samples are at the midpoints.
    if samples == 1
        contour = [pixels1; pixels2];
        lengths12 = (lengths1 + lengths2) * 2 / 3;
        cCCLengths = [lengths12; 2 * lengths12];
        
    % The samples are at the head and tail.
    elseif samples == 2
        contour = pixels1;
        cCCLengths = [length1 / 3; (2 * length1) / 3];
        
    % The samples include, at least the head and tail.
    else
        contour = [pixels1; pixels2(2:(end - 1),:)];
        lengths1 = lengths1 + lengths2(end) - lengths2(end - 1);
        lengths2 = lengths1(end) + lengths2(2:(end - 1));
        cCCLengths = [lengths1; lengths2];
    end
    
    % Compute the contour's head and tail indices.
    cHeadI = 1;
    cTailI = samples;
    
    % Compute the contour angles.
    if isempty(worm)
        
        % The samples are at the midpoints.
        if samples == 1
            cAngles = [0; 0];
            
        % The samples are the head and tail.
        elseif samples == 2
            cAngles = [90; 90];
            
        % The samples divide the worm evenly between its head and tail.
        else
            cAngles = circCurvature(contour, ...
                (cCCLengths(1) + cCCLengths(end)) * (2 / cWormSegs), ...
                cCCLengths);
        end
        
    % Determine the contour angles.
    % The samples are at the midpoints.
    elseif samples == 1
        hLength = worm.contour.chainCodeLengths(worm.contour.headI);
        tLength = worm.contour.chainCodeLengths(worm.contour.tailI);
        eLength = worm.contour.chainCodeLengths(end);
        caLengths = [(hLength + tLength) / 2; ...
            (hLength + tLength + eLength) / 2];
        if caLengths(2) > eLength
            caLengths(2)  = caLengths(2) - eLength;
        end
        cAngles = chainCodeLengthInterp(worm.contour.angles, caLengths, ...
            worm.contour.chainCodeLengths);
        
    % The samples are the head and tail.
    elseif samples == 2
        cAngles = [worm.contour.angles(worm.contour.headI); ...
            worm.contour.angles(worm.contour.tailI)];
        
    % The samples divide the worm evenly between its head and tail.
    else
        caLengths = cCCLengths - cCCLengths(1) + ...
            worm.contour.chainCodeLengths(worm.contour.headI);
        wrap = caLengths > worm.contour.chainCodeLengths(end);
        caLengths(wrap) = caLengths(wrap) - ...
            worm.contour.chainCodeLengths(end);
        cAngles = chainCodeLengthInterp(worm.contour.angles, caLengths, ...
            worm.contour.chainCodeLengths);
    end
    
else % worm.contour.headI > worm.contour.tailI
    
    % The samples are the midpoint.
    if samples == 1
        contour = [pixels2; pixels1];
        lengths12 = (lengths1 + lengths2) * 2 / 3;
        cCCLengths = [lengths12; 2 * lengths12];
        
    % The samples include, at least the head and tail.
    else
        contour = [pixels2; pixels1(2:(end - 1),:)];
        lengths2 = lengths2 + lengths1(end) - lengths1(end - 1);
        lengths1 = lengths2(end) + lengths1(2:(end - 1));
        cCCLengths = [lengths2; lengths1];
    end
    
    % Compute the contour's head and tail indices.
    cHeadI = samples;
    cTailI = 1;
    
    % Determine the contour angles.
    % The samples are at the midpoints.
    if samples == 1
        hLength = worm.contour.chainCodeLengths(worm.contour.headI);
        tLength = worm.contour.chainCodeLengths(worm.contour.tailI);
        eLength = worm.contour.chainCodeLengths(end);
        caLengths = [(hLength + tLength) / 2; ...
            (hLength + tLength + eLength) / 2];
        if caLengths(2) > eLength
            caLengths(2)  = caLengths(2) - eLength;
        end
        cAngles = chainCodeLengthInterp(worm.contour.angles, ...
            caLengths, worm.contour.chainCodeLengths);
        
    % The samples are the head and tail.
    elseif samples == 2
        cAngles = [worm.contour.angles(worm.contour.tailI); ...
            worm.contour.angles(worm.contour.headI)];
        
    % The samples divide the worm evenly between its head and tail.
    else
        caLengths = cCCLengths - cCCLengths(1) + ...
            worm.contour.chainCodeLengths(worm.contour.tailI);
        wrap = caLengths > worm.contour.chainCodeLengths(end);
        caLengths(wrap) = caLengths(wrap) - ...
            worm.contour.chainCodeLengths(end);
        cAngles = chainCodeLengthInterp(worm.contour.angles, ...
            caLengths, worm.contour.chainCodeLengths);
    end
end

% Compute the skeleton's pixels, angles, and widths.
widths = widths / pixel2MicronMagnitude;
if isHeadFlipped
    skeleton = flipud(skeleton);
    skeletonAngles = flipud(skeletonAngles);
    widths = flipud(widths);
end
if isVulvaClockwiseFromHead == isHeadFlipped
    skeletonAngles = -skeletonAngles;
end

% Compute the worm boundaries.
if samples == 1 % The samples are the midpoint.
    
    % Compute the head bounds.
    hsBounds = [1; 1];
    hlcBounds = [2; 2];
    hrcBounds = [1; 1];
    
    % Compute the tail bounds.
    tsBounds = [1; 1];
    tlcBounds = [2; 2];
    trcBounds = [1; 1];
    
    % Compute the left-side bounds.
    lsBounds = [1; 1];
    lcBounds = [2; 2];
    
    % Compute the right-side bounds.
    rsBounds = [1; 1];
    rcBounds = [1; 1];
    
elseif samples == 2 % The samples are the head and tail.
    
    % Compute the head bounds.
    hsBounds = [1; 1];
    hlcBounds = [1; 1];
    hrcBounds = [1; 1];
    
    % Compute the tail bounds.
    tsBounds = [2; 2];
    tlcBounds = [2; 2];
    trcBounds = [2; 2];
    
    % Compute the left-side bounds.
    lsBounds = [1; 1];
    lcBounds = [2; 2];
    
    % Compute the right-side bounds.
    rsBounds = [1; 1];
    rcBounds = [2; 2];
    
else % The samples divide the worm evenly between its head and tail.
    
    % Compute the head and tail segment lengths.
    % Side1 always goes from head to tail in positive, index increments.
    % Side2 always goes from head to tail in negative, index increments.
    htSLength = sCCLengths(end) * (4 / sWormSegs);
    htCLength1 = length1 * (8 / cWormSegs);
    htCLength2 = length2 * (8 / cWormSegs);
    
    % Interpolate the head's skeleton bounds.
    % Note: head/tail flipping is handled later.
    hsBoundLengths = [sCCLengths(1); sCCLengths(1) +  htSLength];
    hsBounds = [1; chainCodeLength2Index(hsBoundLengths(2), sCCLengths)];
    hsBoundPixels = [skeleton(1,:); ...
        round(chainCodeLengthInterp(skeleton, hsBoundLengths(2), ...
        sCCLengths, hsBounds(2)))];
    
    % Interpolate the head's left-side (anti-clockwise) contour bounds.
    % Note: head/tail flipping is handled later.
    hlcBoundLengths = [cCCLengths(cHeadI) - htCLength2; cCCLengths(cHeadI)];
    if hlcBoundLengths(1) < 0
        hlcBoundLengths(1) = hlcBoundLengths(1) + cCCLengths(end);
    end
    hlcBounds = [chainCodeLength2Index(hlcBoundLengths(1), cCCLengths); ...
        cHeadI];
    hlcBoundPixels = [round(chainCodeLengthInterp(contour, ...
        hlcBoundLengths(1), cCCLengths, hlcBounds(1))); ...
        contour(cHeadI, :)];
    
    % The head's left-side (anti-clockwise) inner bound wraps.
    if hlcBoundLengths(1) < cCCLengths(1) || ...
            hlcBoundLengths(1) > cCCLengths(end)
        if hlcBounds(2) == 1
            hlcInBounds = [];
        else
            hlcInBounds = [1; hlcBounds(2) - 1];
        end
        
    % The head's left-side (anti-clockwise) bound is an inner bound.
    elseif  hlcBoundLengths(1) < cCCLengths(hlcBounds(1))
        if hlcBounds(1) ==  hlcBounds(2)
            hlcInBounds = [];
        elseif hlcBounds(2) > 1
            hlcInBounds = [hlcBounds(1); hlcBounds(2) - 1];
        else % wrap
            hlcInBounds = [hlcBounds(1); length(cCCLengths)];
        end
        
    % The head's left-side (anti-clockwise) bound is an outer bound.
    else
        if (hlcBounds(1) == length(cCCLengths) && hlcBounds(2) == 1) ...
                || (hlcBounds(2) - hlcBounds(1) == 1)
            hlcInBounds = [];
        else
            if hlcBounds(1) < length(cCCLengths)
                hlcInBounds(1,1) = hlcBounds(1) + 1;
            else % wrap
                hlcInBounds(1,1) = 1;
            end
            if hlcBounds(2) > 1
                hlcInBounds(2,1) = hlcBounds(2) - 1;
            else % wrap
                hlcInBounds(2,1) = length(cCCLengths);
            end
        end
    end
    
    % Interpolate the head's right-side (clockwise) contour bounds.
    % Note: head/tail flipping is handled later.
    hrcBoundLengths = [cCCLengths(cHeadI); cCCLengths(cHeadI) + htCLength1];
    if hrcBoundLengths(2) > cCCLengths(end)
        hrcBoundLengths(2) = hrcBoundLengths(2) - cCCLengths(end);
    end
    hrcBounds = [cHeadI; chainCodeLength2Index(hrcBoundLengths(2), ...
        cCCLengths)];
    hrcBoundPixels = [contour(cHeadI,:); ...
        round(chainCodeLengthInterp(contour, hrcBoundLengths(2), ...
        cCCLengths, hrcBounds(2)))];
    
    % The head's right-side (clockwise) inner bound wraps.
    if hrcBoundLengths(2) < cCCLengths(1) || ...
            hrcBoundLengths(2) > cCCLengths(end)
        if hrcBounds(1) == length(cCCLengths)
            hrcInBounds = [];
        else
            hrcInBounds = [hrcBounds(1) + 1;  length(cCCLengths)];
        end
        
    % The head's right-side (clockwise) bound is an inner bound.
    elseif  hrcBoundLengths(2) > cCCLengths(hrcBounds(2))
        if hrcBounds(1) ==  hrcBounds(2)
            hrcInBounds = [];
        elseif hrcBounds(1) < length(cCCLengths)
            hrcInBounds = [hrcBounds(1) + 1; hrcBounds(2)];
        else % wrap
            hrcInBounds = [length(cCCLengths); hrcBounds(2)];
        end
        
    % The head's right-side (anti-clockwise) bound is an outer bound.
    else
        if (hrcBounds(1) == length(cCCLengths) && hrcBounds(2) == 1) ...
                || (hrcBounds(2) - hrcBounds(1) == 1)
            hrcInBounds = [];
        else
            if hrcBounds(1) < length(cCCLengths)
                hrcInBounds(1,1) = hrcBounds(1) + 1;
            else % wrap
                hrcInBounds(1,1) = 1;
            end
            if hrcBounds(2) > 1
                hrcInBounds(2,1) = hrcBounds(2) - 1;
            else % wrap
                hrcInBounds(2,1) = length(cCCLengths);
            end
        end
    end
    
    % Interpolate the tail's skeleton bounds.
    % Note: head/tail flipping is handled later.
    tsBoundLengths = [sCCLengths(end) - htSLength; sCCLengths(end)];
    tsBounds = [chainCodeLength2Index(tsBoundLengths(1), sCCLengths); ...
        length(sCCLengths)];
    tsBoundPixels = [round(chainCodeLengthInterp(skeleton, ...
        tsBoundLengths(1), sCCLengths, tsBounds(1))); skeleton(end,:)];
    
    % Interpolate the tail's left-side (anti-clockwise) contour bounds.
    % Note: head/tail flipping is handled later.
    tlcBoundLengths = [cCCLengths(cTailI); cCCLengths(cTailI) + htCLength2];
    if tlcBoundLengths(2) > cCCLengths(end)
        tlcBoundLengths(2) = tlcBoundLengths(2) - cCCLengths(end);
    end
    tlcBounds = [cTailI; chainCodeLength2Index(tlcBoundLengths(2), ...
        cCCLengths)];
    tlcBoundPixels = [contour(cTailI,:); ...
        round(chainCodeLengthInterp(contour, tlcBoundLengths(2), ...
        cCCLengths, tlcBounds(2)))];
    
    % The tail's left-side (clockwise) inner bound wraps.
    if tlcBoundLengths(2) < cCCLengths(1) || ...
            tlcBoundLengths(2) > cCCLengths(end)
        if tlcBounds(1) == length(cCCLengths)
            tlcInBounds = [];
        else
            tlcInBounds = [tlcBounds(1) + 1;  length(cCCLengths)];
        end
        
    % The tail's left-side (clockwise) bound is an inner bound.
    elseif  tlcBoundLengths(2) > cCCLengths(tlcBounds(2))
        if tlcBounds(1) ==  tlcBounds(2)
            tlcInBounds = [];
        elseif tlcBounds(1) < length(cCCLengths)
            tlcInBounds = [tlcBounds(1) + 1; tlcBounds(2)];
        else % wrap
            tlcInBounds = [length(cCCLengths); tlcBounds(2)];
        end
        
    % The tail's left-side (anti-clockwise) bound is an outer bound.
    else
        if (tlcBounds(1) == length(cCCLengths) && tlcBounds(2) == 1) ...
                || (tlcBounds(2) - tlcBounds(1) == 1)
            tlcInBounds = [];
        else
            if tlcBounds(1) < length(cCCLengths)
                tlcInBounds(1,1) = tlcBounds(1) + 1;
            else % wrap
                tlcInBounds(1,1) = 1;
            end
            if tlcBounds(2) > 1
                tlcInBounds(2,1) = tlcBounds(2) - 1;
            else % wrap
                tlcInBounds(2,1) = length(cCCLengths);
            end
        end
    end
    
    % Interpolate the tail's right-side (clockwise) contour bounds.
    % Note: head/tail flipping is handled later.
    trcBoundLengths = [cCCLengths(cTailI) - htCLength1; cCCLengths(cTailI)];
    if trcBoundLengths(1) < 0
        trcBoundLengths(1) = trcBoundLengths(1) + cCCLengths(end);
    end
    trcBounds = [chainCodeLength2Index(trcBoundLengths(1), cCCLengths); ...
        cTailI];
    trcBoundPixels = [round(chainCodeLengthInterp(contour, ...
        trcBoundLengths(1), cCCLengths, trcBounds(1))); contour(cTailI,:)];
    
    % The tail's right-side (anti-clockwise) inner bound wraps.
    if trcBoundLengths(1) < cCCLengths(1) || ...
            trcBoundLengths(1) > cCCLengths(end)
        if trcBounds(2) == 1
            trcInBounds = [];
        else
            trcInBounds = [1; trcBounds(2) - 1];
        end
        
    % The tail's right-side (anti-clockwise) bound is an inner bound.
    elseif  trcBoundLengths(1) < cCCLengths(trcBounds(1))
        if trcBounds(1) ==  trcBounds(2)
            trcInBounds = [];
        elseif trcBounds(2) > 1
            trcInBounds = [trcBounds(1); trcBounds(2) - 1];
        else % wrap
            trcInBounds = [trcBounds(1); length(cCCLengths)];
        end
        
    % The tail's right-side (anti-clockwise) bound is an outer bound.
    else
        if (trcBounds(1) == length(cCCLengths) && trcBounds(2) == 1) ...
                || (trcBounds(2) - trcBounds(1) == 1)
            trcInBounds = [];
        else
            if trcBounds(1) < length(cCCLengths)
                trcInBounds(1,1) = trcBounds(1) + 1;
            else % wrap
                trcInBounds(1,1) = 1;
            end
            if trcBounds(2) > 1
                trcInBounds(2,1) = trcBounds(2) - 1;
            else % wrap
                trcInBounds(2,1) = length(cCCLengths);
            end
        end
    end
    
    % Interpolate the left-side (anti-clockwise) skeleton bounds.
    lsBoundLengths = [hsBoundLengths(2); tsBoundLengths(1)];
    lsBounds = [hsBounds(2); tsBounds(1)];
    lsBoundPixels = [hsBoundPixels(2,:); tsBoundPixels(1,:)];
    
    % Interpolate the left-side (anti-clockwise) skeleton inner bounds.
    if lsBoundLengths(1) < sCCLengths(lsBounds(1))
        lsInBounds(1,1) = lsBounds(1);
    else
        lsInBounds(1,1) = lsBounds(1) + 1;
    end
    if lsBoundLengths(2) > sCCLengths(lsBounds(2))
        lsInBounds(2,1) = lsBounds(2);
    else
        lsInBounds(2,1) = lsBounds(2) - 1;
    end
    if lsInBounds(1) > lsInBounds(2)
        lsInBounds = [];
    end
    
    % Interpolate the left-side (anti-clockwise) contour bounds.
    lcBoundLengths = [tlcBoundLengths(2); hlcBoundLengths(1)];
    lcBounds = [tlcBounds(2); hlcBounds(1)];
    lcBoundPixels = [tlcBoundPixels(2,:); hlcBoundPixels(1,:)];
    
    % Interpolate the left-side (anti-clockwise) contour inner bounds.
    if lcBoundLengths(1) < cCCLengths(1) || ...
            lcBoundLengths(1) > cCCLengths(end) % wrap
        lcInBounds(1,1) = 1;
    elseif  lcBoundLengths(1) < cCCLengths(lcBounds(1))
        lcInBounds(1,1) = lcBounds(1);
    else
        lcInBounds(1,1) = lcBounds(1) + 1;
    end
    if lcBoundLengths(2) < cCCLengths(1) || ...
            lcBoundLengths(2) > cCCLengths(end) % wrap
        lcInBounds(2,1) = length(cCCLengths);
    elseif  lcBoundLengths(2) > cCCLengths(lcBounds(2))
        lcInBounds(2,1) = lcBounds(2);
    else
        lcInBounds(2,1) = lcBounds(2) - 1;
    end
    
    % Interpolate the right-side (clockwise) skeleton (inner) bounds.
    %rsBoundLengths = lsBoundLengths;
    rsBounds = lsBounds;
    rsInBounds = lsInBounds;
    rsBoundPixels = lsBoundPixels;
    
    % Interpolate the right-side (clockwise) contour bounds.
    rcBoundLengths = [hrcBoundLengths(2); trcBoundLengths(1)];
    rcBounds = [hrcBounds(2); trcBounds(1)];
    rcBoundPixels = [hrcBoundPixels(2,:); trcBoundPixels(1,:)];
    
    % Interpolate the right-side (clockwise) contour inner bounds.
    if rcBoundLengths(1) < cCCLengths(1) || ...
            rcBoundLengths(1) > cCCLengths(end) % wrap
        rcInBounds(1,1) = 1;
    elseif  rcBoundLengths(1) < cCCLengths(lcBounds(1))
        rcInBounds(1,1) = rcBounds(1);
    else
        rcInBounds(1,1) = rcBounds(1) + 1;
    end
    if rcBoundLengths(2) < cCCLengths(1) || ...
            rcBoundLengths(2) > cCCLengths(end) % wrap
        rcInBounds(2,1) = length(cCCLengths);
    elseif  rcBoundLengths(2) > cCCLengths(rcBounds(2))
        rcInBounds(2,1) = rcBounds(2);
    else
        rcInBounds(2,1) = rcBounds(2) - 1;
    end
end

% Compute the head, tail, left, and right pixels.
hPixels = [];
tPixels = [];
lPixels = [];
rPixels = [];
if isempty(worm)
    if samples > 2

        % Compute the head's right (clockwise) side.
        hPixels = [];
        if isempty(hrcInBounds)
            hPixels = [hPixels; drawLine(hrcBoundPixels(1,:), ...
                hrcBoundPixels(2,:))];
        else
            hPixels = [hPixels; drawPolyLine(hrcBoundPixels(1,:), ...
                hrcBoundPixels(2,:), contour, hrcInBounds(1), ...
                hrcInBounds(2), 1)];
        end
        
        % Compute the head's right (clockwise) bottom.
        hPixels = [hPixels; ...
            drawLine(hrcBoundPixels(2,:), hsBoundPixels(2,:))];
        
        % Compute the head's left (anti-clockwise) bottom.
        hPixels = [hPixels; ...
            drawLine(hsBoundPixels(2,:), hlcBoundPixels(1,:))];
        
        % Compute the head's left (anti-clockwise) side.
        if isempty(hlcInBounds)
            hPixels = [hPixels; drawLine(hlcBoundPixels(1,:), ...
                hlcBoundPixels(2,:))];
        else
            hPixels = [hPixels; drawPolyLine(hlcBoundPixels(1,:), ...
                hlcBoundPixels(2,:), contour, hlcInBounds(1), ...
                hlcInBounds(2), 1)];
        end
        
        % Compute the tail's left (anti-clockwise) side.
        tPixels = [];
        if isempty(tlcInBounds)
            tPixels = [tPixels; drawLine(tlcBoundPixels(1,:), ...
                tlcBoundPixels(2,:))];
        else
            tPixels = [tPixels; drawPolyLine(tlcBoundPixels(1,:), ...
                tlcBoundPixels(2,:), contour, tlcInBounds(1), ...
                tlcInBounds(2), 1)];
        end
        
        % Compute the tail's left (anti-clockwise) bottom.
        tPixels = [tPixels; ...
            drawLine(tlcBoundPixels(2,:), tsBoundPixels(1,:))];
        
        % Compute the tail's right (clockwise) bottom.
        tPixels = [tPixels; ...
            drawLine(tsBoundPixels(1,:), trcBoundPixels(1,:))];
        
        % Compute the tail's right (clockwise) side.
        if isempty(trcInBounds)
            tPixels = [tPixels; drawLine(trcBoundPixels(1,:), ...
                trcBoundPixels(2,:))];
        else
            tPixels = [tPixels; drawPolyLine(trcBoundPixels(1,:), ...
                trcBoundPixels(2,:), contour, trcInBounds(1), ...
                trcInBounds(2), 1)];
        end
        
        % Compute the left-side's (anti-clockwise) contour side.
        lPixels = [];
        if isempty(lcInBounds)
            lPixels = [lPixels; drawLine(lcBoundPixels(1,:), ...
                lcBoundPixels(2,:))];
        else
            lPixels = [lPixels; drawPolyLine(lcBoundPixels(1,:), ...
                lcBoundPixels(2,:), contour, lcInBounds(1), ...
                lcInBounds(2), 1)];
        end
        
        % Compute the left-side's (anti-clockwise) head side.
        lPixels = [lPixels; ...
            drawLine(lcBoundPixels(2,:), lsBoundPixels(1,:))];
        
        % Compute the left-side's (anti-clockwise) skeleton side.
        if isempty(lsInBounds)
            lPixels = [lPixels; drawLine(lsBoundPixels(1,:), ...
                lsBoundPixels(2,:))];
        else
            lPixels = [lPixels; drawPolyLine(lsBoundPixels(1,:), ...
                lsBoundPixels(2,:), skeleton, lsInBounds(1), ...
                lsInBounds(2), 1)];
        end
        
        % Compute the left-side's (anti-clockwise) tail side.
        lPixels = [lPixels; ...
            drawLine(lsBoundPixels(2,:), lcBoundPixels(1,:))];
        
        % Compute the right-side's (clockwise) contour side.
        rPixels = [];
        if isempty(rcInBounds)
            rPixels = [rPixels; drawLine(rcBoundPixels(1,:), ...
                rcBoundPixels(2,:))];
        else
            rPixels = [rPixels; drawPolyLine(rcBoundPixels(1,:), ...
                rcBoundPixels(2,:), contour, rcInBounds(1), ...
                rcInBounds(2), 1)];
        end
        
        % Compute the right-side's (clockwise) tail side.
        rPixels = [rPixels; ...
            drawLine(rcBoundPixels(2,:), rsBoundPixels(2,:))];
        
        % Compute the right-side's (clockwise) skeleton side.
        if isempty(rsInBounds)
            rPixels = [rPixels; drawLine(rsBoundPixels(2,:), ...
                rsBoundPixels(1,:))];
        else
            rPixels = [rPixels; drawPolyLine(rsBoundPixels(2,:), ...
                rsBoundPixels(1,:), skeleton, rsInBounds(2), ...
                rsInBounds(1), -1)];
        end
        
        % Compute the right-side's (clockwise) head side.
        rPixels = [rPixels; ...
            drawLine(rsBoundPixels(1,:), rcBoundPixels(1,:))];
    end
    
% Determine the head, tail, left, and right pixels.
else
    hPixels = worm.head.pixels;
    tPixels = worm.tail.pixels;
    lPixels = worm.left.pixels;
    rPixels = worm.right.pixels;
end

% Update the areas.
if ~isempty(worm) && isHeadFlipped
    headArea = tailArea / pixel2MicronArea;
    tailArea = headArea / pixel2MicronArea;
else
    headArea = headArea / pixel2MicronArea;
    tailArea = tailArea / pixel2MicronArea;
end
if ~isempty(worm) && worm.orientation.vulva.isClockwiseFromHead == ...
        worm.orientation.head.isFlipped
    leftArea = vulvaArea / pixel2MicronArea;
    rightArea = nonVulvaArea / pixel2MicronArea;
else
    leftArea = nonVulvaArea / pixel2MicronArea;
    rightArea = vulvaArea / pixel2MicronArea;
end

% Determine the head and tail statistics.
if isempty(worm)
    hCDF = [];
    hStdev = 0;
    tCDF = [];
    tStdev = 0;
    lCDF = [];
    lStdev = 0;
    rCDF = [];
    rStdev = 0;
else
    hCDF = worm.head.cdf;
    hStdev = worm.head.stdev;
    tCDF = worm.tail.cdf;
    tStdev = worm.tail.stdev;
    lCDF = worm.left.cdf;
    lStdev = worm.left.stdev;
    rCDF = worm.right.cdf;
    rStdev = worm.right.stdev;
end

% Construct the normalized worm.
% Note: the bounds where interpolated and, therefore, need to be rounded.
worm = worm2struct(frame, contour, [], [], [], ...
    cAngles, cHeadI, cTailI, cCCLengths, ...
    skeleton, [], [], [], [], skeletonAngles, ...
    sCCLengths(end), sCCLengths, widths, ...
    hlcBounds, hrcBounds, hsBounds, hPixels, headArea, hCDF, hStdev, ...
    tlcBounds, trcBounds, tsBounds, tPixels, tailArea, tCDF, tStdev, ...
    lcBounds, lsBounds, lPixels, leftArea, lCDF, lStdev, ...
    rcBounds, rsBounds, rPixels, rightArea, rCDF, rStdev, ...
    isHeadFlipped, hConfidence, tConfidence, ...
    isVulvaClockwiseFromHead, vConfidence, nvConfidence);
end

% Draw a line between two points.
% Note: the start point is included in the line; but, the end point is
% excluded from the line. This avoids overlap when constructing shapes.
function line = drawLine(startPoint, endPoint)
points =  1 + max(abs(round(startPoint - endPoint)));
line = [linspace(startPoint(1), endPoint(1), points); ...
    linspace(startPoint(2), endPoint(2), points)]';
line = round(line(1:(end - 1),:));
end

% Draw a polygonal line between two points and a set of indices that
% connect them.
% Note: the start point is included in the line; but, the end point is
% excluded from the line. This avoids overlap when constructing shapes.
function line = drawPolyLine(startPoint, endPoint, points, startI, ...
    endI, increment)

% Draw the start segment.
line = drawLine(startPoint, points(startI,:));

% Draw the inter-connecting segments.
while startI ~= endI
    nextI = startI + increment;
    if nextI < 1 % wrap
        nextI = size(points,1);
    elseif nextI > size(points,1) % wrap
        nextI = 1;
    end
    line = [line; drawLine(points(startI,:), points(nextI,:))];
    startI = nextI;
end

% Draw the end segment.
line = [line; drawLine(points(endI,:), endPoint)];
end
