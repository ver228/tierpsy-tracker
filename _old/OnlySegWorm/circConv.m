function [c] = circConv(a, b)
%CIRCCONV Convolve the circularly connected vector a with b.
%
%   [C] = CIRCCONV(A, B)
%
%   Inputs:
%       a - a circularly connected vector
%       b - the vector to convolve with a
%
%   Outputs:
%       c - the convolution of the circularly connected vector a with b
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are the inputs vectors.
if ~isvector(a) || ~isvector(b)
  error('circConv:AorBNotVector', 'A and B must be vectors');
end

% Is the convolution vector too long?
if length(a) < length(b)
  warning('circConv:AsmallerThanB', ...
      'A is smaller than B and, therefore, they cannot be convolved');
  c = a;
  return;
end

% Wrap the ends of A and convolve with B.
wrapSize = ceil(length(b) / 2);
wrapA(1:wrapSize) = a((end - wrapSize + 1):end);
wrapA((end + 1):(end + length(a))) = a;
wrapA((end + 1):(end + wrapSize)) = a(1:wrapSize);
wrapA = conv(wrapA, b, 'same');

% Strip away the wrapped ends of A.
c = wrapA((wrapSize + 1):(end - wrapSize));
end

