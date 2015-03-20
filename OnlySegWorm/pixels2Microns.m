function microns = pixels2Microns(origin, pixels, pixel2MicronScale, ...
    rotation)
%PIXELS2MICRONS Convert onscreen pixel coordinates to real-world micron
%   locations.
%
%   MICRONS = PIXELS2MICRONS(ORIGIN, PIXELS, PIXEL2MICRONSCALE, ROTATION)
%
%   Inputs:
%       origin            - the real-world micron origin (stage location)
%                           for the image
%       pixels            - the onscreen pixel coordinates to convert
%       pixel2MicronScale - the scale for converting pixels to microns
%                           (see readPixels2Microns)
%       rotation          - the rotation matrix (see readPixels2Microns)
%
%   Output:
%       microns - the real-world, micron locations
%
% See also READPIXELS2MICRONS, MICRONS2PIXELS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Rotate the pixels.
pixels = (rotation * pixels')';

% Convert the pixels coordinates to micron locations.
microns(:,1) = origin(1) - pixels(:,1) * pixel2MicronScale(1);
microns(:,2) = origin(2) - pixels(:,2) * pixel2MicronScale(2);
end
