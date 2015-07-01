function [worm2 confidence flippedConfidence] = ...
    orientWorm(worm1, worm2, samples, varargin)
%ORIENTWORM Orient worm2 to match worm1's orientation (by setting
%   worm2.orientation.head.isFlipped
%   and worm2.orientation.vulva.isClockwiseFromHead) based on the proximity
%   of samples along the skeleton of both worms.
%
%   Note: the algorithm is a follows,
%   1. Sample both worms along their skeletons at the fractional distances
%      in 'samples'.
%   2. Compute the distance between the skeleton samples from (1) in both
%      orientations.
%   3. Compare the distances from both orientations in (2) and treat the
%      comparison as an indicator function (1 if the distance is smaller,
%      0 otherwise).
%   4. Sum the distance ratios from (3), between both orientations, and
%      treat these sums as the measure of the orientation confidence:
%      unflipped confidence = sum(flipped distances / unflipped distances)
%      flipped confidence = sum(unflipped distances / flipped distances)
%      Note: to avoid dividing by 0, when the denominator is 0 we
%      substitute the numerator instead of the ratio.
%   5. Choose the orientation with the most indicators from (3) and, if
%      both orientations have the same number of indicators, choose the
%      orientation with the most confidence from (4).
%
%   [WORM2 CONFIDENCE FLIPPEDCONFIDENCE] = ORIENTWORM(WORM1, WORM2, SAMPLES)
%
%   [WORM2 CONFIDENCE FLIPPEDCONFIDENCE] = ORIENTWORM(WORM1, WORM2, SAMPLES,
%                                                     VERBOSE)
%
%   Input:
%       worm1   - the reference worm
%       worm2   - the worm to orient relative to the reference
%       samples - the fractional sample locations along the skeleton
%       verbose - verbose mode shows the worms in a figure
%
%   Output:
%       worm2             - the oriented worm (with
%                           worm2.orientation.head.isFlipped and
%                           worm2.orientation.vulva.isClockwiseFromHead
%                           correctly set so that worm1 and worm2 share
%                           the same orientation)
%       confidence        - the confidence measurement for the orientation
%       flippedConfidence - the confidence measurement for the flipped
%                           (opposite) orientation
%
% See also SEGWORM, ORIENTWORMATCENTROID, ORIENTWORMPOSTCOIL,
% HEADTAILMOVEMENTCONFIDENCE
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are we in verbose mode?
if isempty(varargin)
    verbose = false;
else
    verbose = varargin{1};
end

% Sample worm 1.
skeleton1 = worm1.skeleton;
flippedSamples = [];
if worm1.orientation.head.isFlipped
    flippedSamples = 1 - samples;
    s1 = skeleton1.length * flippedSamples;
else
    s1 = skeleton1.length * samples;
end
[p1, s1I] = chainCodeLengthInterp(skeleton1.pixels, s1, ...
    skeleton1.chainCodeLengths);

% Sample worm 2.
skeleton2 = worm2.skeleton;
if worm2.orientation.head.isFlipped
    if isempty(flippedSamples)
        flippedSamples = 1 - samples;
    end
    s2 = skeleton2.length * flippedSamples;
else
    s2 = skeleton2.length * samples;
end
[p2, s2I] = chainCodeLengthInterp(skeleton2.pixels, s2, ...
    skeleton2.chainCodeLengths);

% Compute the distance between worm 1 and 2 in both orientations.
% Note: the skeleton samples are interpolated and, therefore, distances
% less than 1 are likely to be, in reality, either 1 or 0. We set these to
% 0 to avoid computing extreme confidences.
proximity1 = sqrt(sum((p1 - p2) .^ 2, 2));
proximity1(proximity1 < 1) = 0;
proximity2 = sqrt(sum((p1 - flipud(p2)) .^ 2, 2));
proximity2(proximity2 < 1) = 0;

% Compute the indicators for both orientations.
indicators1 = sum(proximity1 < proximity2);
indicators2 = sum(proximity1 > proximity2);

% Compute the confidence for both orientations.
% Note: to avoid dividing by 0, when the magnitude's distance denominator
% is 0 we substitute the numerator's distance instead of the ratio.
magnitudes1 = proximity2 ./ proximity1;
zeroConfidence = (proximity1 < 1);
magnitudes1(zeroConfidence) = proximity2(zeroConfidence);
confidence1 = mean(magnitudes1);
magnitudes2 = proximity1 ./ proximity2;
zeroConfidence = (proximity2 < 1);
magnitudes2(zeroConfidence) = proximity1(zeroConfidence);
confidence2 = mean(magnitudes2);

% Retain the head-to-tail orientation.
if indicators1 > indicators2 || ...
        (indicators1 == indicators2 && confidence1 >= confidence2)
    confidence = confidence1;
    flippedConfidence = confidence2;
    if (verbose)
        isFlipped = false;
    end

% Flip the head-to-tail orientation.
else
    worm2 = flipWormHead(worm2);
    confidence = confidence2;
    flippedConfidence = confidence1;
    if (verbose)
        isFlipped = true;
    end
end

% Flip the vulval side.
if worm2.orientation.vulva.isClockwiseFromHead ~= ...
    worm1.orientation.vulva.isClockwiseFromHead
    worm2 = flipWormVulva(worm2);
end

% Show the worms.
if verbose
    
    % Construct a pattern to identify the head.
    hImg = [1 1 1 1 1; ...
            1 1 1 1 1; ...
            1 1 1 1 1; ...
            1 1 1 1 1; ...
            1 1 1 1 1];
    [hPattern(:,1) hPattern(:,2)] = find(hImg == 1);
    hPattern(:,1) = hPattern(:,1) - ceil(size(hImg, 1) / 2);
    hPattern(:,2) = hPattern(:,2) - ceil(size(hImg, 2) / 2);
    
    % Construct a pattern to identify the vulva.
    vImg = [0 0 1 0 0; ...
            0 1 1 1 0; ...
            1 1 1 1 1; ...
            0 1 1 1 0; ...
            0 0 1 0 0];
    [vPattern(:,1) vPattern(:,2)] = find(vImg == 1);
    vPattern(:,1) = vPattern(:,1) - ceil(size(vImg, 1) / 2);
    vPattern(:,2) = vPattern(:,2) - ceil(size(vImg, 2) / 2);
    
    % Construct the values for the contour and skeleton curvature heat map.
    intensity = .7;
    zeros361 = zeros(361, 1);
    c361(1:361,1) = intensity;
    cRGB = [c361, zeros361, zeros361]; % red
    hRGB = [intensity 0 0]; % red
    vRGB = [intensity 0 0]; % red
    
    % Copy the worms for verbose mode.
    vWorm1 = worm1;
    vWorm2 = worm2;
    
    % Determine the worms' MER (minimum enclosing rectangle).
    % Note: the skeleton can exit the contour.
    wMinX = min(min(vWorm1.contour.pixels(:,2)), ...
        min(vWorm1.skeleton.pixels(:,2)));
    wMaxX = max(max(vWorm1.contour.pixels(:,2)), ...
        max(vWorm1.skeleton.pixels(:,2)));
    wMinY = min(min(vWorm1.contour.pixels(:,1)), ...
        min(vWorm1.skeleton.pixels(:,1)));
    wMaxY = max(max(vWorm1.contour.pixels(:,1)), ...
        max(vWorm1.skeleton.pixels(:,1)));
    wMinX = min([wMinX, min(vWorm2.contour.pixels(:,2)), ...
        min(vWorm2.skeleton.pixels(:,2))]);
    wMaxX = max([wMaxX, max(vWorm2.contour.pixels(:,2)), ...
        max(vWorm2.skeleton.pixels(:,2))]);
    wMinY = min([wMinY, min(vWorm2.contour.pixels(:,1)), ...
        min(vWorm2.skeleton.pixels(:,1))]);
    wMaxY = max([wMaxY, max(vWorm2.contour.pixels(:,1)), ...
        max(vWorm2.skeleton.pixels(:,1))]);
    
    % Minimize the worms.
    vWorm1.contour.pixels(:,1) = vWorm1.contour.pixels(:,1) - wMinY + 3;
    vWorm1.contour.pixels(:,2) = vWorm1.contour.pixels(:,2) - wMinX + 3;
    vWorm1.skeleton.pixels(:,1) = vWorm1.skeleton.pixels(:,1) - wMinY + 3;
    vWorm1.skeleton.pixels(:,2) = vWorm1.skeleton.pixels(:,2) - wMinX + 3;
    vWorm2.contour.pixels(:,1) = vWorm2.contour.pixels(:,1) - wMinY + 3;
    vWorm2.contour.pixels(:,2) = vWorm2.contour.pixels(:,2) - wMinX + 3;
    vWorm2.skeleton.pixels(:,1) = vWorm2.skeleton.pixels(:,1) - wMinY + 3;
    vWorm2.skeleton.pixels(:,2) = vWorm2.skeleton.pixels(:,2) - wMinX + 3;
    
    % Minimize the samples.
    p1(:,1) = round(p1(:,1)) - wMinY + 3;
    p1(:,2) = round(p1(:,2)) - wMinX + 3;
    p2(:,1) = round(p2(:,1)) - wMinY + 3;
    p2(:,2) = round(p2(:,2)) - wMinX + 3;
    
    % Construct the worms' images.
    emptyImg = ones(wMaxY - wMinY + 5, wMaxX - wMinX + 5);
    img1 = overlayWormAngles(emptyImg, vWorm1, cRGB, [], [], ...
        hPattern, hRGB, 1, vPattern, vRGB, 1);
    img2 = overlayWormAngles(emptyImg, vWorm2, cRGB, [], [], ...
        hPattern, hRGB, 1, vPattern, vRGB, 1);
    
    % Overlay the worm images.
    rImg = img2(:,:,1);
    gImg = img1(:,:,1);
    bImg = rImg;
    bImg(gImg == intensity) = intensity;
    p1I = sub2ind(size(rImg), p1(:,1), p1(:,2));
    rImg(p1I) = intensity;
    gImg(p1I) = 0;
    bImg(p1I) = 0;
    p2I = sub2ind(size(gImg), p2(:,1), p2(:,2));
    rImg(p2I) = 0;
    gImg(p2I) = intensity;
    bImg(p2I) = 0;
    rgbImg(:,:,1) = rImg;
    rgbImg(:,:,2) = gImg;
    rgbImg(:,:,3) = bImg;
    
    % Show the overlay.
    figure;
    imshow(rgbImg);
    title(['\color{darkgreen}Worm 2 (frame = ' ...
        num2str(vWorm2.video.frame) ')']);
    ylabel(['\color{red}Worm 1 (frame = ' num2str(vWorm1.video.frame) ')']);
    xlabel({['\color{blue}Orientation confidence: ' num2str(confidence)], ...
        ['\color{orange}Flipped confidence: ' num2str(flippedConfidence)]});
    
    % Show the vectors.
    hold on;
    if isFlipped
        p2 = flipud(p2);
    end
    pDiff = p2 - p1;
    quiver(p1(:,2), p1(:,1), pDiff(:,2), pDiff(:,1), 0, 'b');
end
end
