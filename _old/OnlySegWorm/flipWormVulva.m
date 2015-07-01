function worm = flipWormVulva(worm)
%FLIPWORMVULVA Flip the vulval orientation of the worm.
%
%   WORM = FLIPWORMVULVA(WORM)
%
%   Input:
%       worm - the worm to flip
%
%   Output:
%       worm - the flipped worm
%
%   See also FLIPWORMCELLVULVA, FLIPWORMHEAD, FLIPWORMDATA, WORM2STRUCT,
%   SEGWORM
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Flip the worm's vulval side.
worm.orientation.vulva.isClockwiseFromHead = ...
    ~worm.orientation.vulva.isClockwiseFromHead;
tmp = worm.orientation.vulva.confidence.vulva;
worm.orientation.vulva.confidence.vulva = ...
    worm.orientation.vulva.confidence.nonVulva;
worm.orientation.vulva.confidence.nonVulva = tmp;
end

