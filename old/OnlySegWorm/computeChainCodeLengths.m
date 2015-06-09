function lengths = computeChainCodeLengths(points)
%COMPUTECHAINCODELENGTHS Compute the chain-code length, at each point, for
%   a continuous line of points.
%
%   LENGTHS = COMPUTECHAINCODELENGTHS(POINTS)
%
%   Input:
%       points - the continuous line of points on which to measure the
%                chain-code length
%
%   Output:
%       lengths - the chain-code length at each point
%
% See also CHAINCODELENGTH2INDEX, CIRCCOMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are the points 2 dimensional?
if ndims(points) ~=2 || (size(points, 1) ~= 2 && size(points, 2) ~= 2)
    error('computeChainCodeLengths:PointsNot2D', ...
        'The matrix of points must be 2 dimensional');
end

% Orient the points as a N-by-2 matrix.
isTransposed = false;
if size(points, 2) ~= 2
    points = points';
    isTransposed = true;
end

% Pre-allocate memory.
lengths = double(zeros(size(points, 1), 1));

% Measure the chain code length.
sqrt2 = sqrt(2);
for i = 2:length(lengths)
    
    % Measure the difference between subsequent points.
    dPoints = abs(points(i,:) - points(i - 1,:));
    
    % No change or we walked in a straight line.
    if any(dPoints == 0)
        lengths(i) = lengths(i - 1) + abs(dPoints(1)) + abs(dPoints(2));
        
    % We walked one point diagonally.
    elseif all(dPoints == 1)
        lengths(i) = lengths(i - 1) + sqrt2;
        
    % We walked fractionally or more than one point.
    else
        lengths(i) = lengths(i - 1) + sqrt(sum(dPoints .^ 2));
    end
end
lengthsMex = computeChainCodeLengthsMex(points);
if any(lengths~=lengthsMex)
    disp('bad')
end
% Transpose the lengths.
if isTransposed
    lengths = lengths';
end
end
