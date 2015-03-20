function oImg = overlayWormTouch(img, worm, headRGB, isHeadOpaque, ...
    tailRGB, isTailOpaque, vulvaRGB, isVulvaOpaque, nonVulvaRGB, isNonVulvaOpaque, ...
    cTouchRGB, isCTouchOpaque, cInRGB, isCInOpaque, cOutRGB, isCOutOpaque, ...
    sTouchRGB, isSTouchOpaque, sInRGB, isSInOpaque, sOutRGB, isSOutOpaque, ...
    sInOutRGB, isSInOutOpaque)
%OVERLAYWORMTOUCH Overlay the worm's contour/skeleton touching/inner/outer
% segments onto an image. Label the head/tail, and vulval/non-vulval sides.
%
%   OIMG = OVERLAYWORMTOUCH(IMG, WORM, HEADRGB, ISHEADOPAQUE,
%      TAILRGB, ISTAILOPAQUE, VULVARGB, ISVULVAOPAQUE, NONVULVARGB, ISNONVULVAOPAQUE,
%      CTOUCHRGB, ISCTOUCHOPAQUE, CINRGB, ISCINOPAQUE, COUTRGB, ISCOUTOPAQUE,
%      STOUCHRGB, ISSTOUCHOPAQUE, SINRGB, ISSINOPAQUE, SOUTRGB, ISSOUTOPAQUE,
%      SINOUTRGB, ISSINOUTOPAQUE)
%
%   Inputs:
%       img              - the image on which to overlay the worm information
%       worm             - the worm's information; SEE also SEGWORM, and WORM2STRUCT
%       headRGB          - the color to use for labeling the worm's head
%                          Note: if headRGB is empty, we don't color the
%                          worm's head.
%       isHeadOpaque     - is the head opaque? If the head is opaque its
%                          RGB color is assigned to the worm's head; if the
%                          head is translucent, its RGB values are used to
%                          scale (multiply) the 3 channels of the image
%       tailRGB          - the color to use for labeling the worm's tail
%                          Note: if tailRGB is empty, we don't color the
%                          worm's tail.
%       isTailOpaque     - is the tail opaque? If the tail is opaque its
%                          RGB color is assigned to the worm's tail; if the
%                          tail is translucent, its RGB values are used to
%                          scale (multiply) the 3 channels of the image
%       vulvaRGB         - the color to use for labeling the worm's vulval side
%                          Note: if vulvaRGB is empty, we don't color the
%                          worm's vulval side.
%       isVulvaOpaque    - is the vulval side opaque? If the vulval side
%                          is opaque its RGB color is assigned to the
%                          worm's vulval side; if the vulval side is
%                          translucent, its RGB values are used to
%                          scale (multiply) the 3 channels of the image
%       nonVulvaRGB      - the color to use for labeling the worm's non-vulval side
%                          Note: if nonVulvaRGB is empty, we don't color the
%                          worm's non-vulval side.
%       isNonVulvaOpaque - is the non-vulval side opaque? If the non-vulval side
%                          is opaque its RGB color is assigned to the
%                          worm's non-vulval side; if the non-vulval side is
%                          translucent, its RGB values are used to
%                          scale (multiply) the 3 channels of the image
%       cTouchRGB        - the color to use for labeling the worm's
%                          touching contour segments
%                          Note: if cTouchRGB is empty, we don't color the
%                          worm's touching contour segments.
%       isCTouchOpaque   - are the touching contour segments opaque?
%                          If the segments are opaque their RGB color is
%                          assigned to the worm's touching contour
%                          segments; if the segments are translucent, their
%                          RGB values are used to scale (multiply) the
%                          3 channels of the image
%       cInRGB           - the color to use for labeling the worm's
%                          inner contour segment(s)
%                          Note: if cInRGB is empty, we don't color the
%                          worm's inner contour segment(s).
%       isCInOpaque      - is the inner contour segment(s) opaque?
%                          If the segment(s) is opaque its RGB color is
%                          assigned to the worm's inner contour
%                          segment(s); if the segment(s) is translucent,
%                          its RGB values are used to scale (multiply) the
%                          3 channels of the image
%       cOutRGB          - the color to use for labeling the worm's
%                          outer contour segment(s)
%                          Note: if cOutRGB is empty, we don't color the
%                          worm's outer contour segment(s).
%       isCOutOpaque     - is the outer contour segment(s) opaque?
%                          If the segment(s) is opaque its RGB color is
%                          assigned to the worm's outer contour
%                          segment(s); if the segment(s) is translucent,
%                          its RGB values are used to scale (multiply) the
%                          3 channels of the image
%       sTouchRGB        - the color to use for labeling the worm's
%                          touching skeleton segments
%                          Note: if sTouchRGB is empty, we don't color the
%                          worm's touching skeleton segments.
%       isSTouchOpaque   - are the touching skeleton segments opaque?
%                          If the segments are opaque their RGB color is
%                          assigned to the worm's touching skeleton
%                          segments; if the segments are translucent, their
%                          RGB values are used to scale (multiply) the
%                          3 channels of the image
%       sInRGB           - the color to use for labeling the worm's
%                          inner skeleton segment(s)
%                          Note: if sInRGB is empty, we don't color the
%                          worm's inner skeleton segment(s).
%       isSInOpaque      - is the inner skeleton segment(s) opaque?
%                          If the segment(s) is opaque its RGB color is
%                          assigned to the worm's inner skeleton
%                          segment(s); if the segment(s) is translucent,
%                          its RGB values are used to scale (multiply) the
%                          3 channels of the image
%       sOutRGB          - the color to use for labeling the worm's
%                          outer skeleton segment(s)
%                          Note: if sOutRGB is empty, we don't color the
%                          worm's outer skeleton segment(s).
%       isSOutOpaque     - is the outer skeleton segment(s) opaque?
%                          If the segment(s) is opaque its RGB color is
%                          assigned to the worm's outer skeleton
%                          segment(s); if the segment(s) is translucent,
%                          its RGB values are used to scale (multiply) the
%                          3 channels of the image
%       sInOutRGB        - the color to use for labeling the worm's
%                          dual inner/outer skeleton segment(s)
%                          Note: if sInOutRGB is empty, we don't color the
%                          worm's inner/outer skeleton segment(s).
%       isSInOutOpaque   - is the dual inner/outer skeleton segment(s) opaque?
%                          If the segment(s) is opaque its RGB color is
%                          assigned to the worm's dual inner/outer skeleton
%                          segment(s); if the segment(s) is translucent,
%                          its RGB values are used to scale (multiply) the
%                          3 channels of the image
%
%   Outputs:
%       oImg - an image overlayed with the worm's contour/skeleton
%              touching/inner/outer segments; and, the head/tail, and
%              vulval/non-vulval sides labeled
%
%   SEE also OVERLAYWORMANGLES, SEGWORM, and WORM2STRUCT
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

% Setup the contour and skeleton.
contour = worm.contour.pixels;
skeleton = worm.skeleton.pixels;

% Are the head and tail flipped?
if worm.orientation.head.isFlipped
    head = worm.tail;
    tail = worm.head;
else
    head = worm.head;
    tail = worm.tail;
end

% Overlay the head.
if ~isempty(headRGB) && ~isempty(head.pixels)
    
    % Determine the head's MER (minimum enclosing rectangle).
    hPoly = head.pixels;
    hMinY = min(hPoly(:,1));
    hMinX = min(hPoly(:,2));
    hHeight = max(hPoly(:,1)) - hMinY + 1;
    hWidth = max(hPoly(:,2)) - hMinX + 1;
    
    % Create a mask of the head.
    merHMask = inPolyMask([(hPoly(:,1) - hMinY + 1) ...
        (hPoly(:,2) - hMinX + 1)], [], [hHeight hWidth]);
    
    % Determine the head pixels.
    [merHPixels(:,1) merHPixels(:,2)] = find(merHMask == 1);
    hPixels(:,1) = merHPixels(:,1) + hMinY - 1;
    hPixels(:,2) = merHPixels(:,2) + hMinX - 1;
    
    % Overlay the head.
    hMask = sub2ind(size(img), hPixels(:,1), hPixels(:,2));
    if isHeadOpaque
        img1(hMask) = headRGB(1);
        img2(hMask) = headRGB(2);
        img3(hMask) = headRGB(3);
    else
        img1(hMask) = round(img(hMask) * headRGB(1));
        img2(hMask) = round(img(hMask) * headRGB(2));
        img3(hMask) = round(img(hMask) * headRGB(3));
    end
end

% Overlay the tail.
if ~isempty(tailRGB) && ~isempty(tail.pixels)
    
    % Determine the tail's MER (minimum enclosing rectangle).
    tPoly = tail.pixels;
    tMinY = min(tPoly(:,1));
    tMinX = min(tPoly(:,2));
    tHeight = max(tPoly(:,1)) - tMinY + 1;
    tWidth = max(tPoly(:,2)) - tMinX + 1;
    
    % Create a mask of the tail.
    merTMask = inPolyMask([(tPoly(:,1) - tMinY + 1) ...
        (tPoly(:,2) - tMinX + 1)], [], [tHeight tWidth]);
    
    % Determine the tail pixels.
    [merTPixels(:,1) merTPixels(:,2)] = find(merTMask == 1);
    tPixels(:,1) = merTPixels(:,1) + tMinY - 1;
    tPixels(:,2) = merTPixels(:,2) + tMinX - 1;

    % Overlay the tail.
    tMask = sub2ind(size(img), tPixels(:,1), tPixels(:,2));
    if isTailOpaque
        img1(tMask) = tailRGB(1);
        img2(tMask) = tailRGB(2);
        img3(tMask) = tailRGB(3);
    else
        img1(tMask) = round(img(tMask) * tailRGB(1));
        img2(tMask) = round(img(tMask) * tailRGB(2));
        img3(tMask) = round(img(tMask) * tailRGB(3));
    end
end

% Where is the vulva?
if worm.orientation.vulva.isClockwiseFromHead == ...
        worm.orientation.head.isFlipped
    vulva = worm.left;
    nonVulva = worm.right;
else
    vulva = worm.right;
    nonVulva = worm.left;
end
    
% Overlay the vulval side.
if ~isempty(vulvaRGB) && ~isempty(vulva.pixels)
    
    % Determine the vuval side's MER (minimum enclosing rectangle).
    vPoly = vulva.pixels;
    vMinY = min(vPoly(:,1));
    vMinX = min(vPoly(:,2));
    vHeight = max(vPoly(:,1)) - vMinY + 1;
    vWidth = max(vPoly(:,2)) - vMinX + 1;
    
    % Create a mask of the vuval side.
    merVMask = inPolyMask([(vPoly(:,1) - vMinY + 1) ...
        (vPoly(:,2) - vMinX + 1)], [], [vHeight vWidth]);
    
    % Determine the vuval side's pixels.
    [merVPixels(:,1) merVPixels(:,2)] = find(merVMask == 1);
    vPixels(:,1) = merVPixels(:,1) + vMinY - 1;
    vPixels(:,2) = merVPixels(:,2) + vMinX - 1;

    % Overlay the vulval side.
    vMask = sub2ind(size(img), vPixels(:,1), vPixels(:,2));
    if isVulvaOpaque
        img1(vMask) = vulvaRGB(1);
        img2(vMask) = vulvaRGB(2);
        img3(vMask) = vulvaRGB(3);
    else
        img1(vMask) = round(img(vMask) * vulvaRGB(1));
        img2(vMask) = round(img(vMask) * vulvaRGB(2));
        img3(vMask) = round(img(vMask) * vulvaRGB(3));
    end
end

% Overlay the non-vulval side.
if ~isempty(nonVulvaRGB) && ~isempty(nonVulva.pixels)
    
    % Determine the non-vuval side's MER (minimum enclosing rectangle).
    nvPoly = nonVulva.pixels;
    nvMinY = min(nvPoly(:,1));
    nvMinX = min(nvPoly(:,2));
    nvHeight = max(nvPoly(:,1)) - nvMinY + 1;
    nvWidth = max(nvPoly(:,2)) - nvMinX + 1;
    
    % Create a mask of the non-vuval side.
    merNVMask = inPolyMask([(nvPoly(:,1) - nvMinY + 1) ...
        (nvPoly(:,2) - nvMinX + 1)], [], [nvHeight nvWidth]);
    
    % Determine the non-vuval side's pixels.
    [merNVPixels(:,1) merNVPixels(:,2)] = find(merNVMask == 1);
    nvPixels(:,1) = merNVPixels(:,1) + nvMinY - 1;
    nvPixels(:,2) = merNVPixels(:,2) + nvMinX - 1;

    % Overlay the non-vulval side.
    nvMask = sub2ind(size(img), nvPixels(:,1), nvPixels(:,2));
    if isNonVulvaOpaque
        img1(nvMask) = nonVulvaRGB(1);
        img2(nvMask) = nonVulvaRGB(2);
        img3(nvMask) = nonVulvaRGB(3);
    else
        img1(nvMask) = round(img(nvMask) * nonVulvaRGB(1));
        img2(nvMask) = round(img(nvMask) * nonVulvaRGB(2));
        img3(nvMask) = round(img(nvMask) * nonVulvaRGB(3));
    end
end

% Overlay the contour.
cTouchI = worm.contour.touchI;
if isempty(cTouchI)
    if ~isempty(cOutRGB) && ~isempty(contour)
        cMask = sub2ind(size(img), contour(:,1), contour(:,2));
        if isCOutOpaque
            img1(cMask) = cOutRGB(1);
            img2(cMask) = cOutRGB(2);
            img3(cMask) = cOutRGB(3);
        else
            img1(cMask) = round(img(cMask) * cOutRGB(1));
            img2(cMask) = round(img(cMask) * cOutRGB(2));
            img3(cMask) = round(img(cMask) * cOutRGB(3));
        end
    end

% Overlay the coiled contour.
else
    
    % Overlay the touching contour.
    cLength = size(contour, 1);
    cMask = zeros(cLength, 1);
    if ~isempty(cTouchRGB)
        for i = 1:size(cTouchI, 1)
            
            % Determine the touching contour segment on side 1.
            s1 = cTouchI(i,1);
            e1 = cTouchI(i,2);
            if s1 < e1
                cTouchLength = e1 - s1 + 1;
                cMask(1:cTouchLength) = ...
                    sub2ind(size(img), contour(s1:e1,1), contour(s1:e1,2));
            else % wrap
                cTouchLength = (cLength - s1 + 1) + e1;
                cMask(1:cTouchLength) = ...
                    [sub2ind(size(img), contour(s1:end,1), contour(s1:end,2));
                    sub2ind(size(img), contour(1:e1,1), contour(1:e1,2))];
            end
            
            % Determine the touching contour segment on side 2.
            s2 = cTouchI(i,3);
            e2 = cTouchI(i,4);
            if s2 < e2
                newCTouchLength = cTouchLength + (e2 - s2 + 1);
                cMask((cTouchLength + 1):newCTouchLength) = ...
                    sub2ind(size(img), contour(s2:e2,1), contour(s2:e2,2));
                cTouchLength = newCTouchLength;
            else % wrap
                newCTouchLength = cTouchLength + (cLength - s2 + 2) + e2;
                cMask((cTouchLength + 1):newCTouchLength) = ...
                    [sub2ind(size(img), contour(s2:end,1), contour(s2:end,2));
                    sub2ind(size(img), contour(1:e2,1), contour(1:e2,2))];
                cTouchLength = newCTouchLength;
            end
            
            % Overlay the touching contour segments.
            if isCTouchOpaque
                img1(cMask(1:cTouchLength)) = cTouchRGB(1);
                img2(cMask(1:cTouchLength)) = cTouchRGB(2);
                img3(cMask(1:cTouchLength)) = cTouchRGB(3);
            else
                img1(cMask(1:cTouchLength)) = ...
                    round(img(cMask(1:cTouchLength)) * cTouchRGB(1));
                img2(cMask(1:cTouchLength)) = ...
                    round(img(cMask(1:cTouchLength)) * cTouchRGB(2));
                img3(cMask(1:cTouchLength)) = ...
                    round(img(cMask(1:cTouchLength)) * cTouchRGB(3));
            end
        end
    end
    
    % Overlay the inner contour.
    if ~isempty(cInRGB)
        cInI = worm.contour.inI;
        for i = 1:size(cInI, 1)
            
            % Determine the inner contour segment.
            is = cInI(i,1);
            ie = cInI(i,2);
            if is < ie
                cInLength = ie - is + 1;
                cMask(1:cInLength) = ...
                    sub2ind(size(img), contour(is:ie,1), contour(is:ie,2));
            else % wrap
                cInLength = (cLength - is + 1) + ie;
                cMask(1:cInLength) = ...
                    [sub2ind(size(img), contour(is:end,1), contour(is:end,2));
                    sub2ind(size(img), contour(1:ie,1), contour(1:ie,2))];
            end
            
            % Overlay the inner contour segments.
            if isCInOpaque
                img1(cMask(1:cInLength)) = cInRGB(1);
                img2(cMask(1:cInLength)) = cInRGB(2);
                img3(cMask(1:cInLength)) = cInRGB(3);
            else
                img1(cMask(1:cInLength)) = ...
                    round(img(cMask(1:cInLength)) * cInRGB(1));
                img2(cMask(1:cInLength)) = ...
                    round(img(cMask(1:cInLength)) * cInRGB(2));
                img3(cMask(1:cInLength)) = ...
                    round(img(cMask(1:cInLength)) * cInRGB(3));
            end
        end
    end
    
    % Overlay the outer contour.
    if ~isempty(cOutRGB)
        cOutI = worm.contour.outI;
        for i = 1:size(cOutI, 1)
            
            % Determine the inner contour segment.
            os = cOutI(i,1);
            oe = cOutI(i,2);
            if is < ie
                cOutLength = oe - os + 1;
                cMask(1:cOutLength) = ...
                    sub2ind(size(img), contour(os:oe,1), contour(os:oe,2));
            else % wrap
                cOutLength = (cLength - os + 1) + oe;
                cMask(1:cOutLength) = ...
                    [sub2ind(size(img), contour(os:end,1), contour(os:end,2));
                    sub2ind(size(img), contour(1:oe,1), contour(1:oe,2))];
            end
            
            % Overlay the outer contour segments.
            if isCOutOpaque
                img1(cMask(1:cOutLength)) = cOutRGB(1);
                img2(cMask(1:cOutLength)) = cOutRGB(2);
                img3(cMask(1:cOutLength)) = cOutRGB(3);
            else
                img1(cMask(1:cOutLength)) = ...
                    round(img(cMask(1:cOutLength)) * cOutRGB(1));
                img2(cMask(1:cOutLength)) = ...
                    round(img(cMask(1:cOutLength)) * cOutRGB(2));
                img3(cMask(1:cOutLength)) = ...
                    round(img(cMask(1:cOutLength)) * cOutRGB(3));
            end
        end
    end
end

% Overlay the skeleton.
sTouchI = worm.skeleton.touchI;
if isempty(sTouchI)
    if ~isempty(sOutRGB) && ~isempty(skeleton)
        sMask = sub2ind(size(img), skeleton(:,1), skeleton(:,2));
        if isSOutOpaque
            img1(sMask) = sOutRGB(1);
            img2(sMask) = sOutRGB(2);
            img3(sMask) = sOutRGB(3);
        else
            img1(sMask) = round(img(sMask) * sOutRGB(1));
            img2(sMask) = round(img(sMask) * sOutRGB(2));
            img3(sMask) = round(img(sMask) * sOutRGB(3));
        end
    end

% Overlay the coiled skeleton.
else
    
    % Overlay the touching skeleton.
    sLength = size(skeleton, 1);
    sMask = zeros(sLength, 1);
    if ~isempty(sTouchRGB)
        for i = 1:size(sTouchI, 1)
            
            % Determine the touching skeleton segment on side 1.
            s1 = sTouchI(i,1);
            e1 = sTouchI(i,2);
            if s1 < e1
                sTouchLength = e1 - s1 + 1;
                sMask(1:sTouchLength) = ...
                    sub2ind(size(img), skeleton(s1:e1,1), skeleton(s1:e1,2));
            else % wrap
                sTouchLength = (sLength - s1 + 1) + e1;
                sMask(1:sTouchLength) = ...
                    [sub2ind(size(img), skeleton(s1:end,1), skeleton(s1:end,2));
                    sub2ind(size(img), skeleton(1:e1,1), skeleton(1:e1,2))];
            end
            
            % Determine the touching skeleton segment on side 2.
            s2 = sTouchI(i,3);
            e2 = sTouchI(i,4);
            if s2 < e2
                newSTouchLength = sTouchLength + (e1 - s1 + 1);
                sMask((sTouchLength + 1):newSTouchLength) = ...
                    sub2ind(size(img), skeleton(s2:e2,1), skeleton(s2:e2,2));
                sTouchLength = newSTouchLength;
            else % wrap
                newSTouchLength = sTouchLength + (sLength - s1 + 1) + e1;
                sMask((sTouchLength + 1):newSTouchLength) = ...
                    [sub2ind(size(img), skeleton(s2:end,1), skeleton(s2:end,2));
                    sub2ind(size(img), skeleton(1:e2,1), skeleton(1:e2,2))];
                sTouchLength = newSTouchLength;
            end
            
            % Overlay the touching skeleton segments.
            if isSTouchOpaque
                img1(sMask(1:sTouchLength)) = sTouchRGB(1);
                img2(sMask(1:sTouchLength)) = sTouchRGB(2);
                img3(sMask(1:sTouchLength)) = sTouchRGB(3);
            else
                img1(sMask(1:sTouchLength)) = ...
                    round(img(sMask(1:sTouchLength)) * sTouchRGB(1));
                img2(sMask(1:sTouchLength)) = ...
                    round(img(sMask(1:sTouchLength)) * sTouchRGB(2));
                img3(sMask(1:sTouchLength)) = ...
                    round(img(sMask(1:sTouchLength)) * sTouchRGB(3));
            end
        end
    end
    
    % Overlay the inner skeleton.
    if ~isempty(sInRGB)
        sInI = worm.skeleton.inI;
        for i = 1:size(sInI, 1)
            
            % Determine the inner skeleton segment.
            is = sInI(i,1);
            ie = sInI(i,2);
            if is < ie
                sInLength = ie - is + 1;
                sMask(1:sInLength) = ...
                    sub2ind(size(img), skeleton(is:ie,1), skeleton(is:ie,2));
            else % wrap
                sInLength = (sLength - is + 1) + ie;
                sMask(1:sInLength) = ...
                    [sub2ind(size(img), skeleton(is:end,1), skeleton(is:end,2));
                    sub2ind(size(img), skeleton(1:ie,1), skeleton(1:ie,2))];
            end
            
            % Overlay the inner skeleton segments.
            if isSInOpaque
                img1(sMask(1:sInLength)) = sInRGB(1);
                img2(sMask(1:sInLength)) = sInRGB(2);
                img3(sMask(1:sInLength)) = sInRGB(3);
            else
                img1(sMask(1:sInLength)) = ...
                    round(img(sMask(1:sInLength)) * sInRGB(1));
                img2(sMask(1:sInLength)) = ...
                    round(img(sMask(1:sInLength)) * sInRGB(2));
                img3(sMask(1:sInLength)) = ...
                    round(img(sMask(1:sInLength)) * sInRGB(3));
            end
        end
    end
    
    % Overlay the outer skeleton.
    if ~isempty(sOutRGB)
        sOutI = worm.skeleton.outI;
        for i = 1:size(sOutI, 1)
            
            % Determine the inner skeleton segment.
            os = sOutI(i,1);
            oe = sOutI(i,2);
            if is < ie
                sOutLength = oe - os + 1;
                sMask(1:sOutLength) = ...
                    sub2ind(size(img), skeleton(os:oe,1), skeleton(os:oe,2));
            else % wrap
                sOutLength = (sLength - os + 1) + oe;
                sMask(1:sOutLength) = ...
                    [sub2ind(size(img), skeleton(os:end,1), skeleton(os:end,2));
                    sub2ind(size(img), skeleton(1:oe,1), skeleton(1:oe,2))];
            end
            
            % Overlay the outer skeleton segments.
            if isSOutOpaque
                img1(sMask(1:sOutLength)) = sOutRGB(1);
                img2(sMask(1:sOutLength)) = sOutRGB(2);
                img3(sMask(1:sOutLength)) = sOutRGB(3);
            else
                img1(sMask(1:sOutLength)) = ...
                    round(img(sMask(1:sOutLength)) * sOutRGB(1));
                img2(sMask(1:sOutLength)) = ...
                    round(img(sMask(1:sOutLength)) * sOutRGB(2));
                img3(sMask(1:sOutLength)) = ...
                    round(img(sMask(1:sOutLength)) * sOutRGB(3));
            end
        end
    end
    
    % Overlay the dual inner/outer skeleton.
    if ~isempty(sInOutRGB)
        sInOutI = worm.skeleton.inOutI;
        for i = 1:size(sInOutI, 1)
            
            % Determine the inner skeleton segment.
            ios = sInOutI(i,1);
            ioe = sInOutI(i,2);
            if ios < ioe
                sInOutLength = ioe - ios + 1;
                sMask(1:sInOutLength) = ...
                    sub2ind(size(img), skeleton(ios:ioe,1), skeleton(ios:ioe,2));
            else % wrap
                sInOutLength = (sLength - ios + 1) + ioe;
                sMask(1:sInOutLength) = ...
                    [sub2ind(size(img), skeleton(ios:end,1), skeleton(ios:end,2));
                    sub2ind(size(img), skeleton(1:ioe,1), skeleton(1:ioe,2))];
            end
            
            % Overlay the outer skeleton segments.
            if isSInOutOpaque
                img1(sMask(1:sInOutLength)) = sInOutRGB(1);
                img2(sMask(1:sInOutLength)) = sInOutRGB(2);
                img3(sMask(1:sInOutLength)) = sInOutRGB(3);
            else
                img1(sMask(1:sInOutLength)) = ...
                    round(img(sMask(1:sInOutLength)) * sInOutRGB(1));
                img2(sMask(1:sInOutLength)) = ...
                    round(img(sMask(1:sInOutLength)) * sInOutRGB(2));
                img3(sMask(1:sInOutLength)) = ...
                    round(img(sMask(1:sInOutLength)) * sInOutRGB(3));
            end
        end
    end
end

% Combine the rgb channels.
oImg(:,:,1) = img1;
oImg(:,:,2) = img2;
oImg(:,:,3) = img3;
end
