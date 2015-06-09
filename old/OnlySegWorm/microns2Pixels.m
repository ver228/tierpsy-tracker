function pixels = microns2Pixels(origin, microns, pixel2MicronScale, ...
    rotation)
%PIXELS2MICRONS Convert real-world micron locations to onscreen pixel
%   coordinates.
%
%   PIXELS = MICRONS2PIXELS(ORIGIN, MICRONS, PIXEL2MICRONSCALE, ROTATION)
%
%   Inputs:
%       origin            - the real-world micron origin (stage location)
%                           for the image
%       microns           - the real-world, micron locations to convert
%       pixel2MicronScale - the scale for converting pixels to microns
%                           (see readPixels2Microns)
%       rotation          - the rotation matrix (see readPixels2Microns)
%
%   Output:
%       pixels - the onscreen pixel coordinates
%
% See also READPIXELS2MICRONS, PIXELS2MICRONS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Convert the micron locations to pixels coordinates.
pixels(:,1) = (origin(1) - microns(:,1)) / pixel2MicronScale(1);
pixels(:,2) = (origin(2) - microns(:,2)) / pixel2MicronScale(2);

% Unrotate the pixels.
if size(rotation,1) == 2 && size(rotation,1) == 2
    rotation(1,2) = -rotation(1,2);
    rotation(2,1) = -rotation(2,1);
else
    rotation = 1 / rotation;
end
pixels = (rotation * pixels')';
end
