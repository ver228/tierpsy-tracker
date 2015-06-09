function [bwImg] = otsuImg(img, varargin)
%OTSUIMG Use the Otsu threshold to binarize an image.
%
%   [BWIMG] = OTSUIMG(IMG)
%
%   [BWIMG] = OTSUIMG(IMG, ISNORMALIZED)
%
%   [BWIMG] = OTSUIMG(IMG, ISNORMALIZED, STDDEV)
%
%   Input:
%       img          - the image to binarize
%       isNormalized - is the image already normalized (i.e., all pixel
%                      values are between 0 to 1, inclusive)?
%       stdDev       - the standard deviations, from the Otsu threshold's
%                      foreground mean, at which to threshold;
%                      if empty, we threshold at the Otsu threshold.
%
%   Output:
%       bwImg - the otsu-threshold, binarized image
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Should we normalize the image?
isNormalized = false;
if ~isempty(varargin)
    isNormalized = varargin{1};
end

% At how many standard deviations, from the Otsu threshold's foreground
% mean, should we threshold?
stdDev = [];
if length(varargin) > 1
    stdDev = varargin{2};
end

% Convert the image to grayscale.
if (size(img,3) == 3)
    img = rgb2gray(img);
end

% Normalize the image.
if ~isNormalized
    img = double(img - min(min(img)));
    img = img / max(max(img));
end

% Compute the Otsu threshold.
thr = graythresh(img);

% Adjust the threshold.
if ~isempty(stdDev)
    if isinteger(img)
        img = double(img) / 255.0;
    end
    foreground = img(img < thr);
    foreground = double(foreground(:));
    thr = mean(foreground) + stdDev * std(foreground);
end

% Binarize the image.
bwImg = im2bw(img, thr);
end
