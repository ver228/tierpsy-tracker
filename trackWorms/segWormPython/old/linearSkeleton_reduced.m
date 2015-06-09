%function [skeleton, cWidths] = linearSkeleton(headI, tailI, minP, minI, ...
%    maxP, maxI, contour, wormSegSize, varargin)

minP = lfCMinP; minI = lfCMinI;
maxP = lfCMaxP; maxI = lfCMaxI; 
wormSegSize = wormSegLength; 
chainCodeLengths = cCCLengths; %= varargin{1};

%LINEARSKELETON Skeletonize a linear (non-looped) worm. The worm is
%skeletonized by splitting its contour, from head to tail, into short
%segments. These short segments are bounded by matching pairs of minimal
%angles (< -20 degrees) and their nearest points on the opposite side of
%the worm's contour. We then walk along the opposing sides of these short
%segments and mark the midline as our skeleton. The final step is cleaning
%up this skeleton to remove overlapping points and interpolate missing ones.
%
%   [SKELETON CWIDTHS] = LINEARSKELETON(HEADI, TAILI, CONTOUR, WORMSEGSIZE)
%
%   [SKELETON CWIDTHS] = LINEARSKELETON(HEADI, TAILI, CONTOUR, WORMSEGSIZE,
%                                       CHAINCODELENGTHS)
%
%   Inputs:
%       headI            - the head's contour index
%       tailI            - the tail's contour index
%       minP             - the local minimal peaks
%       minI             - the local minimal peaks' contour indices
%       minP             - the local maximal peaks
%       minI             - the local maximal peaks' contour indices
%       contour          - the worm's contour
%       wormSegSize      - the size (in contour length) of a worm segment.
%                          Note 1: the worm is roughly divided into 24
%                          segments of musculature (i.e., hinges that
%                          represent degrees of freedom) on each side.
%                          Therefore, 48 segments around a 2-D contour.
%                          Note 2: "In C. elegans the 95 rhomboid-shaped
%                          body wall muscle cells are arranged as staggered
%                          pairs in four longitudinal bundles located in
%                          four quadrants. Three of these bundles (DL, DR,
%                          VR) contain 24 cells each, whereas VL bundle
%                          contains 23 cells." - www.wormatlas.org
%       chainCodeLengths - the chain-code length at each point;
%                          if empty, the array indices are used instead
%
%   Output:
%       skeleton - the worm's skeleton
%       cWidths  - the worm contour's width at each skeleton point
%
% See also SEGWORM, CIRCCOMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software;
% you must reproduce all copyright notices and other proprietary
% notices on any copies of the Software.



% Compute the edge size to use in searching for opposing contour points.
% We use 1/4 of a contour side to be safe.
% Note: worm curvature can significantly distort the length of a contour
% side and, consequently, the locations of identical spots on opposing
% sides of the contour. Therefore, in addition to using scaled locations,
% we also use a large search window to ensure we correctly identify
% opposing contour locations.
searchEdgeSize = chainCodeLengths(end) / 8;

% Compute the segment size to use in excluding the head and tail angles.
% Due to bends and obscure boundaries at the head and tail, it is difficult
% to match opposing contour points near these locations.The worm's head and
% tail occupy, approximately, 4 muscle segments each, on the skeleton and
% either side of the contour.
% Note: "The first two muscle cells in the two ventral and two dorsal rows
% [of the head] are smaller than their lateral counterparts, giving a
% stagger to the packing of the two rows of cells in a quadrant. The first
% four muscles in each quadrant are innervated exclusively by motoneurons
% in the nerve ring. The second block of four muscles is dually innervated,
% receiving synaptic input from motoneurons in the nerve ring and the
% anterior ventral cord. The rest of the muscles in the body are
% exclusively innervated by NMJs in the dorsal and ventral cords." - The
% Structure of the Nervous System of the Nematode C. elegans, on
% www.wormatlas.org
htSegSize = wormSegSize * 4;

% Find small head boundaries.
sHeadI = chainCodeLengths(headI) - htSegSize;
if sHeadI < chainCodeLengths(1)
    sHeadI = sHeadI + chainCodeLengths(end);
end
sHeadI = chainCodeLength2Index(sHeadI, chainCodeLengths);

eHeadI = chainCodeLengths(headI) + htSegSize;
if eHeadI > chainCodeLengths(end)
    eHeadI = eHeadI - chainCodeLengths(end);
end
eHeadI = chainCodeLength2Index(eHeadI, chainCodeLengths);

% Find small tail boundaries.
sTailI = chainCodeLengths(tailI) - htSegSize;
if sTailI < chainCodeLengths(1)
    sTailI = sTailI + chainCodeLengths(end);
end
sTailI = chainCodeLength2Index(sTailI, chainCodeLengths);
eTailI = chainCodeLengths(tailI) + htSegSize;
if eTailI > chainCodeLengths(end)
    eTailI = eTailI - chainCodeLengths(end);
end
eTailI = chainCodeLength2Index(eTailI, chainCodeLengths);

% Find the large minimal bends away from the head and tail.
bendI = [minI(minP < -20); maxI(maxP > 20)];
bendI(betweenPoints(bendI, sHeadI, eHeadI)) = [];
bendI(betweenPoints(bendI, sTailI, eTailI)) = [];

% Compute the head, tail, midbody, and bends for both sides.
% Skeletonization occurs piecemeal, stitching together segments starting at
% the midbody, from bend to bend, and ending with the head/tail.
% Side1 always goes from head to tail in positive, index increments.
% Side2 always goes from head to tail in negative, index increments.
cLength = size(contour, 1);
if headI <= tailI
    
    % Compute the head indices.
    h1 = headI + 1;
    h2 = headI - 1;
    if h2 < 1
        h2 = h2 + cLength;
    end
    
    % Compute the tail indices.
    t1 = tailI - 1;
    t2 = tailI + 1;
    if t2 > cLength
        t2 = t2 - cLength;
    end
    %%
    % Compute the midbody indices for side 1.
    m1s1 = chainCodeLength2Index((chainCodeLengths(headI) + ...
        chainCodeLengths(tailI)) / 2, chainCodeLengths);
    m1s2 = circOpposingNearestPoints(m1s1, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    %%
    % Compute the midbody indices for side 2.
    m2s2 = (chainCodeLengths(headI) + chainCodeLengths(tailI) + ...
        chainCodeLengths(end)) / 2;
    if m2s2 > chainCodeLengths(end)
        m2s2 = m2s2 - chainCodeLengths(end);
    end
    m2s2 = chainCodeLength2Index(m2s2, chainCodeLengths);
    m2s1 = circOpposingNearestPoints(m2s2, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % The closest points are the true midbody indices.
    if sum((contour(m1s1,:) - contour(m1s2,:)) .^ 2) <= ...
            sum((contour(m2s1,:) - contour(m2s2,:)) .^ 2)
        m1 = m1s1;
        m2 = m1s2;
    else
        m1 = m2s1;
        m2 = m2s2;
    end
    
    % Compute the minimum distance between the midbody indices.
    if m1 > m2
        dM = min(m1 - m2, m2 + size(contour, 1) - m1);
    else
        dM = min(m2 - m1, m1 + size(contour, 1) - m2);
    end
    
    % Compute the bend indices for side 1.
    bendI1 = bendI >= headI & bendI <= tailI;
    b1 = bendI(bendI1);
    oppB1 = circOpposingNearestPoints(b1, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Remove any bend indices too close to the head and/or tail.
    headB1 = betweenPoints(oppB1, sHeadI, eHeadI);
    b1(headB1) = [];
    oppB1(headB1) = [];
    tailB1 = betweenPoints(oppB1, sTailI, eTailI);
    b1(tailB1) = [];
    oppB1(tailB1) = [];
    
    % Remove any bend indices that cross the midline.
    crossed = maxDistPoints(b1, oppB1, dM, contour);
    b1(crossed) = [];
    oppB1(crossed) = [];
    
    % Minimize the width at the bend.
    b1 = circOpposingNearestPoints(oppB1, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Remove any bend indices too close to the head and/or tail.
    headB1 = betweenPoints(b1, sHeadI, eHeadI);
    b1(headB1) = [];
    oppB1(headB1) = [];
    tailB1 = betweenPoints(b1, sTailI, eTailI);
    b1(tailB1) = [];
    oppB1(tailB1) = [];
    
    % Remove any bend indices that cross the midline.
    crossed = maxDistPoints(b1, oppB1, dM, contour);
    b1(crossed) = [];
    oppB1(crossed) = [];
    
    % Compute the bend indices for side 2.
    b2 = bendI(~bendI1);
    oppB2 = circOpposingNearestPoints(b2, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Remove any bend indices too close to the head and/or tail.
    headB2 = betweenPoints(oppB2, sHeadI, eHeadI);
    b2(headB2) = [];
    oppB2(headB2) = [];
    tailB2 = betweenPoints(oppB2, sTailI, eTailI);
    b2(tailB2) = [];
    oppB2(tailB2) = [];
    
    % Remove any bend indices that cross the midline.
    crossed = maxDistPoints(b2, oppB2, dM, contour);
    b2(crossed) = [];
    oppB2(crossed) = [];
    
    % Minimize the width at the bend.
    b2 = circOpposingNearestPoints(oppB2, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Remove any bend indices too close to the head and/or tail.
    headB2 = betweenPoints(b2, sHeadI, eHeadI);
    b2(headB2) = [];
    oppB2(headB2) = [];
    tailB2 = betweenPoints(b2, sTailI, eTailI);
    b2(tailB2) = [];
    oppB2(tailB2) = [];
    
    % Remove any bend indices that cross the midline.
    crossed = maxDistPoints(b2, oppB2, dM, contour);
    b2(crossed) = [];
    oppB2(crossed) = [];
    
    % Combine the bend indices from opposing sides and order everything so
    % that the skeleton segments can never cross.
    [b1 bO] = sort([m1; b1; oppB2]);
    b2 = sort([m2; b2; oppB1], 1, 'descend');
    b2 = [b2(b2 <= headI); b2(b2 >= tailI)];
    mI = find(bO == 1);
    
    % Compute the inter-bend indices.
    ib1 = chainCodeLength2Index((chainCodeLengths(b1(1:(end - 1))) + ...
        chainCodeLengths(b1(2:end))) / 2, chainCodeLengths);
    ib2 = zeros(length(ib1), 1);
    for i = 1:length(ib2)
        ib2(i) = circNearestPoints(contour(ib1(i),:), b2(i + 1), b2(i), ...
            contour);
    end
    
    % Side 1 wraps.
else % headI > tailI
    
    % Compute the head indices.
    h1 = headI + 1;
    if h1 > cLength
        h1 = h1 - cLength;
    end
    h2 = headI - 1;
    
    % Compute the tail indices.
    t1 = tailI - 1;
    if t1 < 1
        t1 = t1 + cLength;
    end
    t2 = tailI + 1;
    
    % Compute the midbody indices for side 2.
    m1s2 = chainCodeLength2Index((chainCodeLengths(headI) + ...
        chainCodeLengths(tailI)) / 2, chainCodeLengths);
    m1s1 = circOpposingNearestPoints(m1s2, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Compute the midbody indices for side 1.
    m2s1 = (chainCodeLengths(headI) + chainCodeLengths(tailI) + ...
        chainCodeLengths(end)) / 2;
    if m2s1 > chainCodeLengths(end)
        m2s1 = m2s1 - chainCodeLengths(end);
    end
    m2s1 = chainCodeLength2Index(m2s1, chainCodeLengths);
    m2s2 = circOpposingNearestPoints(m2s1, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % The closest points are the true midbody indices.
    if sum((contour(m1s1,:) - contour(m1s2,:)) .^ 2) <= ...
            sum((contour(m2s1,:) - contour(m2s2,:)) .^ 2)
        m1 = m1s1;
        m2 = m1s2;
    else
        m1 = m2s1;
        m2 = m2s2;
    end
    
    % Compute the minimum distance between the midbody indices.
    if m1 > m2
        dM = min(m1 - m2, m2 + size(contour, 1) - m1);
    else
        dM = min(m2 - m1, m1 + size(contour, 1) - m2);
    end
    
    % Compute the bend indices for side 2.
    bendI2 = bendI <= headI & bendI >= tailI;
    b2 = bendI(bendI2);
    oppB2 = circOpposingNearestPoints(b2, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Remove any bend indices too close to the head and/or tail.
    headB2 = betweenPoints(oppB2, sHeadI, eHeadI);
    b2(headB2) = [];
    oppB2(headB2) = [];
    tailB2 = betweenPoints(oppB2, sTailI, eTailI);
    b2(tailB2) = [];
    oppB2(tailB2) = [];
    
    % Remove any bend indices that cross the midline.
    crossed = maxDistPoints(b2, oppB2, dM, contour);
    b2(crossed) = [];
    oppB2(crossed) = [];
    
    % Minimize the width at the bend.
    b2 = circOpposingNearestPoints(oppB2, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Remove any bend indices too close to the head and/or tail.
    headB2 = betweenPoints(b2, sHeadI, eHeadI);
    b2(headB2) = [];
    oppB2(headB2) = [];
    tailB2 = betweenPoints(b2, sTailI, eTailI);
    b2(tailB2) = [];
    oppB2(tailB2) = [];
    
    % Remove any bend indices that cross the midline.
    crossed = maxDistPoints(b2, oppB2, dM, contour);
    b2(crossed) = [];
    oppB2(crossed) = [];
    
    % Compute the bend indices for side 1.
    b1 = bendI(~bendI2);
    oppB1 = circOpposingNearestPoints(b1, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Remove any bend indices too close to the head and/or tail.
    headB1 = betweenPoints(oppB1, sHeadI, eHeadI);
    b1(headB1) = [];
    oppB1(headB1) = [];
    tailB1 = betweenPoints(oppB1, sTailI, eTailI);
    b1(tailB1) = [];
    oppB1(tailB1) = [];
    
    % Remove any bend indices that cross the midline.
    crossed = maxDistPoints(b1, oppB1, dM, contour);
    b1(crossed) = [];
    oppB1(crossed) = [];
    
    % Minimize the width at the bend.
    b1 = circOpposingNearestPoints(oppB1, contour, headI, tailI, ...
        searchEdgeSize, chainCodeLengths);
    
    % Remove any bend indices too close to the head and/or tail.
    headB1 = betweenPoints(b1, sHeadI, eHeadI);
    b1(headB1) = [];
    oppB1(headB1) = [];
    tailB1 = betweenPoints(b1, sTailI, eTailI);
    b1(tailB1) = [];
    oppB1(tailB1) = [];
    
    % Remove any bend indices that cross the midline.
    crossed = maxDistPoints(b1, oppB1, dM, contour);
    b1(crossed) = [];
    oppB1(crossed) = [];
    
    % Combine the bend indices from opposing sides and order everything so
    % that the skeleton segments can never cross.
    b1 = sort([m1; b1; oppB2]);
    b1 = [b1(b1 >= headI); b1(b1 <= tailI)];
    [b2 bO] = sort([m2; b2; oppB1], 1, 'descend');
    mI = find(bO == 1);
    
    % Compute the inter-bend indices.
    ib2 = chainCodeLength2Index((chainCodeLengths(b2(1:(end - 1))) + ...
        chainCodeLengths(b2(2:end))) / 2, chainCodeLengths);
    ib1 = zeros(length(ib2), 1);
    for i = 1:length(ib2)
        ib1(i) = circNearestPoints(contour(ib2(i),:), b1(i), b1(i + 1), ...
            contour);
    end
end

% Skeletonize the worm from its midbody to its head.
mhSkeleton = zeros(size(contour, 1), 2);
mhWidths = zeros(size(contour, 1), 1);
i = mI;
j = 1;
while i > 1
    
    % Skeletonize the segment from the bend to the interbend.
    %[segment widths] = skeletonize(b1(i), ib1(i - 1), -1, ...
    %    b2(i), ib2(i - 1), 1, contour, contour, false);
    
    [segment, widths] = skeletonizeMex(b1(i), ib1(i - 1), -1, ...
        b2(i), ib2(i - 1), 1, contour, contour);
    
    nextJ = j + size(segment, 1) - 1;
    mhSkeleton(j:nextJ,:) = segment;
    mhWidths(j:nextJ) = widths;
    j = nextJ + 1;
    i = i - 1;
    
    % Skeletonize the segment from the next bend back to the interbend.
    %[segment widths] = skeletonize(b1(i), ib1(i), 1, ...
    %    b2(i), ib2(i), -1, contour, contour, false);
    [segment, widths] = skeletonizeMex(b1(i), ib1(i), 1, ...
        b2(i), ib2(i), -1, contour, contour);
    
    nextJ = j + size(segment, 1) - 1;
    mhSkeleton(j:nextJ,:) = flipud(segment);
    mhWidths(j:nextJ) = flipud(widths);
    j = nextJ + 1;
end

% Skeletonize the segment from the last bend to the head.
%[segment widths] = skeletonize(b1(i), h1, -1, b2(i), h2, 1, ...
%    contour, contour, false);
[segment, widths] = skeletonizeMex(b1(i), h1, -1, b2(i), h2, 1, ...
    contour, contour);
nextJ = j + size(segment, 1) - 1;
mhSkeleton(j:nextJ,:) = segment;
mhWidths(j:nextJ) = widths;

% Clean up.
mhSkeleton((nextJ + 1):end,:) = [];
mhWidths((nextJ + 1):end) = [];

% Skeletonize the worm from its midbody to its tail.
mtSkeleton = zeros(size(contour, 1), 2);
mtWidths = zeros(size(contour, 1), 1);
i = mI;
j = 1;
while i < length(b1)
    
    % Skeletonize the segment from the bend to the interbend.
    %[segment widths] = skeletonize(b1(i), ib1(i), 1, ...
    %    b2(i), ib2(i), -1, contour, contour, false);
    [segment, widths] = skeletonizeMex(b1(i), ib1(i), 1, ...
        b2(i), ib2(i), -1, contour, contour);
    nextJ = j + size(segment, 1) - 1;
    mtSkeleton(j:nextJ,:) = segment;
    mtWidths(j:nextJ) = widths;
    j = nextJ + 1;
    i = i + 1;
    
    % Skeletonize the segment from the next bend back to the interbend.
    %[segment widths] = skeletonize(b1(i), ib1(i - 1), -1, ...
    %    b2(i), ib2(i - 1), 1, contour, contour, false);
    [segment,widths] = skeletonizeMex(b1(i), ib1(i - 1), -1, ...
        b2(i), ib2(i - 1), 1, contour, contour);
    nextJ = j + size(segment, 1) - 1;
    mtSkeleton(j:nextJ,:) = flipud(segment);
    mtWidths(j:nextJ) = flipud(widths);
    j = nextJ + 1;
end

% Skeletonize the segment from the last bend to the tail.
%[segment, widths] = skeletonize(b1(i), t1, 1, b2(i), t2, -1, ...
%    contour, contour, false);
[segment, widths] = skeletonizeMex(b1(i), t1, 1, b2(i), t2, -1, ...
    contour, contour);
nextJ = j + size(segment, 1) - 1;
mtSkeleton(j:nextJ,:) = segment;
mtWidths(j:nextJ) = widths;

% Clean up.
mtSkeleton((nextJ + 1):end,:) = [];
mtWidths((nextJ + 1):end) = [];

% % Skeletonize the worm from its midbody to its head and tail.
% [mhSkeleton mhWidths] = skeletonize(m1, h1, -1, m2, h2, 1, ...
%     contour, contour, false);
% [mtSkeleton mtWidths] = skeletonize(m1, t1, 1, m2, t2, -1, ...
%     contour, contour, false);

% Reconstruct the skeleton.
skeleton = [contour(headI,:); flipud(mhSkeleton); mtSkeleton; contour(tailI,:)];
cWidths = [0; flipud(mhWidths); mtWidths; 0];

% Clean up the rough skeleton.
skeleton = round(skeleton);
%oSkeleton = skeleton; ocWidths = cWidths;
%[skeleton, cWidths] = cleanSkeletonMex(oSkeleton, ocWidths, wormSegSize);
[skeleton, cWidths] = cleanSkeletonMex(skeleton, cWidths, wormSegSize);



