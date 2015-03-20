function worm = flipWormHead(worm)
%FLIPWORMHEAD Flip the head-to-tail orientation of the worm.
%
%Note: since the vulva is specified relative to the head, its location
%flips to preserve its orientation.
%
%   WORM = FLIPWORMHEAD(WORM)
%
%   Input:
%       worm - the worm to flip
%
%   Output:
%       worm - the flipped worm
%
%   See also FLIPWORMCELLHEAD, FLIPWORMVULVA, FLIPWORMDATA, WORM2STRUCT,
%   SEGWORM
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Flip the worm's head and tail.
worm.orientation.head.isFlipped = ~worm.orientation.head.isFlipped;
tmp = worm.orientation.head.confidence.head;
worm.orientation.head.confidence.head = ...
    worm.orientation.head.confidence.tail;
worm.orientation.head.confidence.tail = tmp;

% Flip the worm's vulval side.
worm.orientation.vulva.isClockwiseFromHead = ...
    ~worm.orientation.vulva.isClockwiseFromHead;
end

