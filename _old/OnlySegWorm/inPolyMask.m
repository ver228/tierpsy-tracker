function [mask polygonI] = inPolyMask(polygon, seed, maskSize)
%INPOLYMASK Create a mask of the inside of a polygon.
%
%   MASK = INPOLYMASK(POLYGON, CENTER, MASKSIZE)
%
%   Inputs:
%       polygon  - the polygon whose inside to mask
%       seed     - a seed point inside the polygon;
%                  if empty, a slower, hole-filling algorithm is used
%       maskSize - the size of the mask
%
%   Output:
%       mask     - a (logical) mask of the inside of the polygon
%       polygonI - the polygon indices for the mask
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Create the mask.
mask = false(maskSize);

% Fill in the polygon.
polygonI = sub2ind(maskSize, polygon(:,1), polygon(:,2));
mask(polygonI) = true;
if isempty(seed)
    mask = imfill(mask, 'holes');
else
    mask = imfill(mask, seed);
end

% Remove the polygon's perimeter.
mask(polygonI) = false;
end

