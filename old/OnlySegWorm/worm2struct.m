function worm = worm2struct(frame, ...
    cPixels, cTouchI, cInI, cOutI, cAngles, cHeadI, cTailI, cCCLengths, ...
    sPixels, sTouchI, sInI, sOutI, sInOutI, sAngles, ...
    sLength, sCCLengths, sWidths, ...
    hlcBounds, hrcBounds, hsBounds, hPixels, hArea, hCDF, hStdev, ...
    tlcBounds, trcBounds, tsBounds, tPixels, tArea, tCDF, tStdev, ...
    lcBounds, lsBounds, lPixels, lArea, lCDF, lStdev, ...
    rcBounds, rsBounds, rPixels, rArea, rCDF, rStdev, ...
    isHeadTailFlipped, hConfidence, tConfidence,...
    isVulvaClockwiseFromHead, vConfidence, nvConfidence)
%WORM2STRUCT Organize worm information in a structure.
%
%   WORM = WORM2STRUCT(FRAME,
%       CPIXELS, CTOUCHI, CINI, COUTI, CANGLES, CHEADI, CTAILI,
%       SPIXELS, STOUCHI, SINI, SOUTI, SINOUTI, SANGLES, SLENGTH, SWIDTHS,
%       HLCBOUNDS, HRCBOUNDS, HSBOUNDS, HPIXELS, HCDF, HSTDEV,
%       TLCBOUNDS, TRCBOUNDS, TSBOUNDS, TPIXELS, TCDF, TSTDEV,
%       LCBOUNDS, LSBOUNDS, LPIXELS, LCDF, LSTDEV,
%       RCBOUNDS, RSBOUNDS, RPIXELS, RCDF, RSTDEV
%       ISHEADTAILFLIPPED, ISVULVACLOCKWISEFROMHEAD,
%       HCONFIDENCE, TCONFIDENCE, VCONFIDENCE, NVCONFIDENCE)
%
%   Inputs:
%
%       * Contour *
%
%       cPixels    - the worm's circularly continuous contour pixels,
%                    ordered clockwise
%       cTouchI    - the paired pairs of indices marking, clockwise, the
%                    start and end of the touching contour points
%                    Note: if the worm isn't coiled, this value is empty.
%       cInI       - the paired indices marking, clockwise, the start and
%                    end of the inner contour points
%                    Note: if the worm isn't coiled, this value is empty.
%       cOutI      - the paired indices marking, clockwise, the start and
%                    end of the outer contour points
%                    Note: if the worm isn't coiled, this value is empty.
%       cAngles    - the contour's angles (curvature) at each index
%       cHeadI     - the contour index for the worm's head
%       cTailI     - the contour index for the worm's tail
%       cCCLengths - the contour's circular chain-code pixel length, from
%                    its vector's start to end, up to each contour point
%                    Note: this is a more accurate representation of
%                    locations along the worm's contour than pixel indices
%
%       * Skeleton *
%
%       sPixels    - the worm's continuous skeleton oriented from head to tail
%       sTouchI    - the paired pairs of indices marking, clockwise, the
%                    start and end of the touching skeleton points
%                    Note: if the worm isn't coiled, this value is empty.
%       sInI       - the paired indices marking, clockwise, the start and
%                    end of the inner skeleton points
%                    Note: if the worm isn't coiled, this value is empty.
%       sOutI      - the paired indices marking, clockwise, the start and
%                    end of the outer skeleton points
%                    Note: if the worm isn't coiled, this value is empty.
%       sInOutI    - the pairs of indices marking, from head to tail,
%                    the start and end of the dual inner/outer skeleton points
%                    Note: if the worm isn't coiled, this value is empty.
%       sAngles    - the skeleton's angles (curvature) per point
%                    Note 1: NaNs indicate the end pieces where there is
%                    insufficient information to compute the angle
%                    Note 2: positive skeleton angles bulge towards the side
%                    clockwise from the worm's head (unless the worm is flipped)
%       sLength    - the skeleton's (worm's) chain-code pixel length
%       sCCLengths - the skeleton's (worm's) chain-code pixel length, from
%                    head to tail, up to each skeleton point
%                    Note: this is a more accurate representation of
%                    locations along the worm's skeleton than pixel indices
%       sWidths    - the contour's (worm's) widths, from head to tail, at
%                    each skeleton point
%
%       * Head *
%
%       hlcBounds - the worm head's, left-side (counter clockwise from the head),
%                   contour bounds (the start and end indices of the segment)
%       hrcBounds - the worm head's, right-side (clockwise from the head),
%                   contour bounds (the start and end indices of the segment)
%       hsBounds  - the worm head's, skeleton bounds (the start and end
%                   indices of the segment)
%                   Note: due to the clockwise ordering of the worm contour
%                   and the head-to-tail ordering of the worm skeleton,
%                   the bounds enclose the head as
%                   [hsBounds(1), hrcBounds(1:2), hsBounds(2), hlcBounds(1:2)]
%       hPixels   - the worm head's circularly continuous contour pixels
%       hArea     - the worm head's pixel area
%       hCDF      - the worm head's pixel-intensity, cumulative distribution
%                   function at 2.5%, 25%, 50%, 75%, and 97.5%
%       hStdev    - the worm head's pixel-intensity standard deviation
%
%       * Tail *
%
%       tlcBounds - the worm tail's, left-side (counter clockwise from the head),
%                   contour bounds (the start and end indices of the segment)
%       trcBounds - the worm tail's, right-side (clockwise from the head),
%                   contour bounds (the start and end indices of the segment)
%       tsBounds  - the worm tail's, skeleton bounds (the start and end
%                   indices of the segment)
%                   Note: due to the clockwise ordering of the worm contour
%                   and the head-to-tail ordering of the worm skeleton,
%                   the bounds enclose the tail as
%                   [tsBounds(1), trcBounds(1:2), tsBounds(2), tlcBounds(1:2)]
%       tPixels   - the worm tail's circularly continuous contour pixels
%       tArea     - the worm tail's pixel area
%       tCDF      - the worm tail's pixel-intensity, cumulative distribution
%                   function at 2.5%, 25%, 50%, 75%, and 97.5%
%       tStdev    - the worm tail's pixel-intensity standard deviation
%
%       * Left Side (Counter Clockwise from the Head) *
%
%       lcBounds - the worm's, left-side (counter clockwise from the head),
%                  contour bounds (the start and end indices of the segment)
%       lsBounds - the worm's, left-side (counter clockwise from the head),
%                  skeleton bounds (the start and end indices of the segment)
%                  Note: due to the clockwise ordering of the worm contour
%                  and the head-to-tail ordering of the worm skeleton,
%                  the bounds enclose the left side as
%                  [lcBounds(1:2), lsBounds(2:1)]
%       lPixels  - the worm's left-side (counter clockwise from the head)
%                 circularly continuous contour pixels
%       lArea    - the worm's left-side pixel area
%       lCDF     - the worm's left-side (counter clockwise from the head)
%                  pixel-intensity, cumulative distribution function at
%                  2.5%, 25%, 50%, 75%, and 97.5%
%       lStdev   - the worm's left-side (counter clockwise from the head)
%                  pixel-intensity standard deviation
%
%       * Right Side (Clockwise from the Head) *
%
%       rcBounds - the worm's, right-side (clockwise from the head),
%                  contour bounds (the start and end indices of the segment)
%       rsBounds - the worm's, right-side (clockwise from the head),
%                  skeleton bounds (the start and end indices of the segment)
%                  Note: due to the clockwise ordering of the worm contour
%                  and the head-to-tail ordering of the worm skeleton,
%                  the bounds enclose the left side as
%                  [rcBounds(1:2), rsBounds(2:1)]
%       rPixels  - the worm's right-side (clockwise from the head)
%                  circularly continuous contour pixels
%       rArea    - the worm's right-side pixel area
%       rCDF     - the worm's right-side (clockwise from the head)
%                  pixel-intensity, cumulative distribution function at
%                  2.5%, 25%, 50%, 75%, and 97.5%
%       rStdev   - the worm's right-side (clockwise from the head)
%                  pixel-intensity standard deviation
%
%       * Orientation *
%
%       isHeadTailFlipped        - are the head and tail flipped?
%                                  Note 1: the head and tail may be
%                                  incorrectly assigned. This flag, allows
%                                  the assignment to be easily flipped.
%                                  Note 2: this flag also flips the
%                                  skeleton orientation.
%       hConfidence              - how much confidence do we have in our
%                                  head choice as the worm's head?
%       tConfidence              - how much confidence do we have in our
%                                  tail choice as the worm's head?
%       isVulvaClockwiseFromHead - is the vulva on the side clockwise from
%                                  the head?
%       vConfidence              - how much confidence do we have in our
%                                  vulval-side choice as the worm's
%                                  vulval side?
%       nvConfidence             - how much confidence do we have in our
%                                  non-vulval-side choice as the worm's
%                                  vulval side?
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
%   See also WORM2CELL, CELL2WORM, SEGWORM
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Organize the video information.
video = struct('frame', frame);

% Organize the contour.
contour = struct('pixels', cPixels, 'touchI', cTouchI, ...
    'inI', cInI, 'outI', cOutI, 'angles', cAngles, ...
    'headI', cHeadI, 'tailI', cTailI, 'chainCodeLengths', cCCLengths);

% Organize the skeleton.
skeleton = struct('pixels', sPixels, ...
    'touchI', sTouchI, 'inI', sInI, 'outI', sOutI, 'inOutI', sInOutI, ...
    'angles', sAngles, 'length', sLength, ...
    'chainCodeLengths', sCCLengths, 'widths', sWidths);

% Organize the head.
hContour = struct('left', hlcBounds, 'right', hrcBounds);
hBounds = struct('contour', hContour, 'skeleton', hsBounds);
head = struct('bounds', hBounds, 'pixels', hPixels, 'area', hArea, ...
    'cdf', hCDF, 'stdev', hStdev);

% Organize the tail.
tContour = struct('left', tlcBounds, 'right', trcBounds);
tBounds = struct('contour', tContour, 'skeleton', tsBounds);
tail = struct('bounds', tBounds, 'pixels', tPixels, 'area', tArea, ...
    'cdf', tCDF, 'stdev', tStdev);

% Organize the worm's left side (counter clockwise from the head) .
lBounds = struct('contour', lcBounds, 'skeleton', lsBounds);
left = struct('bounds', lBounds, 'pixels', lPixels, 'area', lArea, ...
    'cdf', lCDF, 'stdev', lStdev);

% Organize the worm's right side (clockwise from the head) .
rBounds = struct('contour', rcBounds, 'skeleton', rsBounds);
right = struct('bounds', rBounds, 'pixels', rPixels, 'area', rArea, ...
    'cdf', rCDF, 'stdev', rStdev);

% Organize the worm's orientation.
headConfidence = struct('head', hConfidence, 'tail', tConfidence);
headOrientation = struct('isFlipped', isHeadTailFlipped, ...
    'confidence', headConfidence);
vulvaConfidence = struct('vulva', vConfidence, 'nonVulva', nvConfidence);
vulvaOrientation = struct('isClockwiseFromHead', isVulvaClockwiseFromHead, ...
    'confidence', vulvaConfidence);
orientation = struct('head', headOrientation, 'vulva', vulvaOrientation);

% Organize the worm.
worm = struct('video', video, 'contour', contour, 'skeleton', skeleton, ...
    'head', head, 'tail', tail, 'left', left, 'right', right, ...
    'orientation', orientation);
end
