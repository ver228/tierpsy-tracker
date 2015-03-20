function oImg = overlayWormAngles(img, worm, cRGB360, sRGB360, sRGBNaN, ...
    headPattern, headRGB, isHeadOpaque, vulvaPattern, vulvaRGB, isVulvaOpaque)
%OVERLAYWORMANGLES Overlay the worm's contour/skeleton angles (curvature)
% onto an image. Label the head and vulval side.
%
%   OIMG = OVERLAYWORMANGLES(IMG, WORM, CRGB360, SRGB360, SRGBNAN,
%      HEADPATTERN, HEADRGB, ISHEADOPAQUE, VULVAPATTERN, VULVARGB, ISVULVAOPAQUE)
%
%   Inputs:
%       img           - the image on which to overlay the worm information
%       worm          - the worm's information; SEE also SEGWORM, and WORM2STRUCT
%       cRGB360       - the color map for the worm's contour angles (use
%                       361 RGB values to be safe); SEE also COLORMAP
%                       Note: if cRGB360 is empty, we don't color the
%                       worm's contour angles.
%       sRGB360       - the color map for the worm's skeleton angles (use
%                       361 RGB values to be safe); SEE also COLORMAP
%                       Note: if sRGB360 is empty, we don't color the
%                       worm's skeleton angles.
%       sRGBNaN       - the color to use for the worm's undefned (NaN) skeleton angles
%                       Note: if sRGBNaN is empty, we don't color the
%                       worm's undefined (NaN) skeleton angles.
%       headPattern   - the pattern to use for labeling the worm's head;
%                       the pattern is a 2-by-N matrix of the row and
%                       column offsets (centered at the head) to color
%                       Note: if headPattern is empty, we don't label the
%                       worm's head.
%       headRGB       - the color(s) to use for labeling the worm's head;
%                       if only one RGB value is present, the head pattern
%                       is uniformly colored; if multiple RGB values are
%                       present, each head pattern pixel is assigned its
%                       corresponding value
%       isHeadOpaque  - is the head opaque? If the head is opaque its RGB
%                       colors are assigned to the head pattern; if the
%                       head is translucent, its RGB values are used to
%                       scale (multiply) the 3 channels of the image
%       vulvaPattern  - the pattern to use for labeling the worm's vulva;
%                       the pattern is a 2-by-N matrix of the row and
%                       column offsets (centered at the vulva) to color
%                       Note: if vulvaPattern is empty, we don't label the
%                       worm's vulva.
%       vulvaRGB      - the color(s) to use for labeling the worm's vulva;
%                       if only one RGB value is present, the vulva pattern
%                       is uniformly colored; if multiple RGB values are
%                       present, each vulva pattern pixel is assigned its
%                       corresponding value
%       isVulvaOpaque - is the vulva opaque? If the vulva is opaque its RGB
%                       colors are assigned to the vulva pattern; if the
%                       vulva is translucent, its RGB values are used to
%                       scale (multiply) the 3 channels of the image
%
%   Outputs:
%       oImg - an image overlayed with the worm's contour/skeleton angles
%              (curvature); and, the head and vulval side labeled
%
%   SEE also OVERLAYWORMTOUCH, SEGWORM, and WORM2STRUCT
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Convert the image to grayscale.
if (size(img,3) == 3)
    img = rgb2gray(img);
end

% Setup the rgb channels.
img1 = img;
img2 = img;
img3 = img;

% Initialize the contour and skeleton.
contour = worm.contour.pixels;
skeleton = worm.skeleton.pixels;
    
% Overlay the head.
if ~isempty(headPattern)
    
    % Are the head and tail flipped?
    if worm.orientation.head.isFlipped
        head = worm.tail;
    else
        head = worm.head;
    end
    
    % Compute the center of the head.
    hlcBounds = head.bounds.contour.left;
    hrcBounds = head.bounds.contour.right;
    hCenter = circPolyCenter(hlcBounds(1), hlcBounds(2), contour, ...
        hrcBounds(1), hrcBounds(2), contour, ...
        worm.contour.chainCodeLengths, worm.contour.chainCodeLengths);
    
    % Construct the head pattern.
    headPattern(:,1) = headPattern(:,1) + hCenter(1);
    headPattern(:,2) = headPattern(:,2) + hCenter(2);
    headPattern(headPattern(:,1) < 1, :) = [];
    headPattern(headPattern(:,1) > size(img, 1), :) = [];
    headPattern(headPattern(:,2) < 1, :) = [];
    headPattern(headPattern(:,2) > size(img, 2), :) = [];
    headPatternI = sub2ind(size(img), headPattern(:,1),  headPattern(:,2));

    % Overlay the head.
    if isHeadOpaque
        if size(headRGB, 1) > 1
            img1(headPatternI) = headRGB(1,:);
            img2(headPatternI) = headRGB(2,:);
            img3(headPatternI) = headRGB(3,:);
        else
            img1(headPatternI) = headRGB(1);
            img2(headPatternI) = headRGB(2);
            img3(headPatternI) = headRGB(3);
        end
    else
        if size(headRGB, 1) > 1
            img1(headPatternI) = round(img1(headPatternI) .* headRGB(1,:));
            img2(headPatternI) = round(img2(headPatternI) .* headRGB(2,:));
            img3(headPatternI) = round(img3(headPatternI) .* headRGB(3,:));
        else
            img1(headPatternI) = round(img1(headPatternI) * headRGB(1));
            img2(headPatternI) = round(img2(headPatternI) * headRGB(2));
            img3(headPatternI) = round(img3(headPatternI) * headRGB(3));
        end
    end
end

% Overlay the vulva.
if ~isempty(vulvaPattern)
    
    % Where is the vulva?
    if worm.orientation.vulva.isClockwiseFromHead == ...
            worm.orientation.head.isFlipped
        vulva = worm.left;
    else
        vulva = worm.right;
    end
    
    % Compute the center of the vulval side.
    vcBounds = vulva.bounds.contour;
    vsBounds = vulva.bounds.skeleton;
    vCenter = circPolyCenter(vcBounds(1), vcBounds(2), contour, ...
        vsBounds(1), vsBounds(2), skeleton, ...
        worm.contour.chainCodeLengths, worm.skeleton.chainCodeLengths);
    
    % Construct the vulva pattern.
    vulvaPattern(:,1) = vulvaPattern(:,1) + vCenter(1);
    vulvaPattern(:,2) = vulvaPattern(:,2) + vCenter(2);
    vulvaPattern(vulvaPattern(:,1) < 1, :) = [];
    vulvaPattern(vulvaPattern(:,1) > size(img, 1), :) = [];
    vulvaPattern(vulvaPattern(:,2) < 1, :) = [];
    vulvaPattern(vulvaPattern(:,2) > size(img, 2), :) = [];
    vulvaPatternI = sub2ind(size(img), vulvaPattern(:,1),  vulvaPattern(:,2));
    
    % Overlay the vulva.
    if isVulvaOpaque
        if size(vulvaRGB, 1) > 1
            img1(vulvaPatternI) = vulvaRGB(1,:);
            img2(vulvaPatternI) = vulvaRGB(2,:);
            img3(vulvaPatternI) = vulvaRGB(3,:);
        else
            img1(vulvaPatternI) = vulvaRGB(1);
            img2(vulvaPatternI) = vulvaRGB(2);
            img3(vulvaPatternI) = vulvaRGB(3);
        end
    else
        if size(vulvaRGB, 1) > 1
            img1(vulvaPatternI) = round(img1(vulvaPatternI) .* vulvaRGB(1,:));
            img2(vulvaPatternI) = round(img2(vulvaPatternI) .* vulvaRGB(2,:));
            img3(vulvaPatternI) = round(img3(vulvaPatternI) .* vulvaRGB(3,:));
        else
            img1(vulvaPatternI) = round(img1(vulvaPatternI) * vulvaRGB(1));
            img2(vulvaPatternI) = round(img2(vulvaPatternI) * vulvaRGB(2));
            img3(vulvaPatternI) = round(img3(vulvaPatternI) * vulvaRGB(3));
        end
    end
end

% Overlay the contour angles.
if ~isempty(cRGB360) && ~isempty(worm.contour.angles)
    cAngles = worm.contour.angles;
    cAngles360 = round(cAngles + 181);
    cImgMask = sub2ind(size(img), contour(:,1), contour(:,2));
    img1(cImgMask) = cRGB360(cAngles360, 1);
    img2(cImgMask) = cRGB360(cAngles360, 2);
    img3(cImgMask) = cRGB360(cAngles360, 3);
end

% Overlay the skeleton.
if ~isempty(sRGB360) && ~isempty(worm.skeleton.angles)

    % Orient the skeleton angles with the vulval side.
    if worm.orientation.vulva.isClockwiseFromHead == ...
            worm.orientation.head.isFlipped
        sAngles = -worm.skeleton.angles;
    else
        sAngles = worm.skeleton.angles;
    end
    
    % Overlay the skeleton angles.
    nanMask = isnan(sAngles);
    anglesMask = ~nanMask;
    sAngles360 = round(sAngles(anglesMask) + 181);
    sImgMask = sub2ind(size(img), skeleton(:,1), skeleton(:,2));
    sAnglesImgMask = sImgMask(anglesMask);
    img1(sAnglesImgMask) = sRGB360(sAngles360, 1);
    img2(sAnglesImgMask) = sRGB360(sAngles360, 2);
    img3(sAnglesImgMask) = sRGB360(sAngles360, 3);
    
    % Overlay the skeleton NaN angles.
    if ~isempty(sRGBNaN)
        sNaNImgMask = sImgMask(nanMask);
        img1(sNaNImgMask) = sRGBNaN(1);
        img2(sNaNImgMask) = sRGBNaN(2);
        img3(sNaNImgMask) = sRGBNaN(3);
    end
end

% Combine the rgb channels.
oImg(:,:,1) = img1;
oImg(:,:,2) = img2;
oImg(:,:,3) = img3;
end
