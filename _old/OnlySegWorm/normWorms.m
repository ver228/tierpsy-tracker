function [vulvaContours nonVulvaContours skeletons angles inOutTouches ...
    lengths widths headAreas tailAreas vulvaAreas nonVulvaAreas isNormed] = ...
    normWorms(worms, samples, moves, origins, pixel2MicronScale, rotation, verbose)
moves = [0, 0];
origins = [0,0];
pixel2MicronScale = [1, 1];
rotation = 1;
verbose = false;

%NORMWORMS Normalize the worms' information to a standard, compact set.
%
%   [VULVACONTOURS NONVULVACONTOURS SKELETONS ANGLES TOUCHINOUTS LENGTHS
%      WIDTHS HEADAREAS TAILAREAS VULVAAREAS NONVULVAAREAS ISNORMED] =
%      NORMWORMS(WORMS, SAMPLES, MOVES, ORIGINS, PIXEL2MICRONSCALE,
%      ROTATION, VERBOSE)
%
%   Inputs:
%       worms             - the worms to normalize (see segWorm)
%       samples           - the number of samples to take
%                           Note 1: the worm information is downsampled
%                           to fit normally and compactly, for example,
%                           within a database.
%                           Note 2: we recommend downsampling to 65 points.
%                           65 points oversample a wild-type C. elegans,
%                           providing, at least, the Nyquist rate of twice
%                           its degrees of freedom. Moreover, 65 points
%                           assigns points to the head, midbody, and tail
%                           with 62 points left over to assign midpoints at
%                           the 1/4, 1/8, 1/16, and 1/32 worm locations.
%       moves             - a 2-D matrix with, respectively, the start and
%                           end frame indices of stage movements
%                           (see findStageMovements, movesI)
%       origins           - the real-world micron origins (stage locations)
%                           for the worms
%                           (see findStageMovements, locations)
%       pixel2MicronScale - the scale for converting pixels to microns
%                           (see readPixels2Microns, pixel2MicronScale)
%       rotation          - the rotation matrix for onscreen pixels
%                           (see readPixels2Microns, rotation)
%       verbose           - verbose mode shows the results in a figure
%
%   Outputs:
%
%       Note 1: all outputs are oriented head to tail.
%       Note 2: all coordinates are oriented as (x,y); in contrast, in the
%       worm structure, coordinates are oriented as (row,column) = (y,x).
%
%       vulvaContours    - the worms' downsampled vulval-side contours
%       nonVulvaContours - the worms' downsampled non-vulval-side contours
%       skeletons        - the worms' downsampled skeletons
%       angles           - the worms' downsampled skeleton angles
%                          Note: positive skeleton angles bulge towards the
%                          vulva; in contrast, in the worm structure,
%                          positive skeleton angles bulge towards the side
%                          clockwise from the worm's head (unless the worm
%                          is flipped).
%       inOutTouches     - in coiled worms, for each skeleton sample:
%                          -1 = the contours are inside the coil
%                          -2 = the contours are outside the coil
%                          -3 = the contours are both inside and outside
%                               the coil (one half of the contour is inside
%                               the coil, the other half of the contour is
%                               outside it)
%                          1+ = the contour is touching itself to form the
%                               coil; the specific number represents the
%                               contraposed skeleton index being touched
%       lengths          - the worms' skeletons' chain-code pixel lengths
%       widths           - the worms' contours downsampled widths
%       headAreas        - the worms' head areas
%       tailAreas        - the worms' tail areas
%       vulvaAreas       - the worms' vulval-side areas
%                          (excluding the head and tail)
%       nonVulvaAreas    - the worms' non-vulval-side areas
%                          (excluding the head and tail)
%       isNormed         - was the worm normalized?
%                          (worms with insufficient sampling points,
%                          dropped frames, stage movements, and
%                          segmentation failures are marked as false)
%
% See also SEGWORM, FINDSTAGEMOVEMENT, READPIXELS2MICRONS, NORM2WORM
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Pre-compute values.
pixel2MicronArea = sum(pixel2MicronScale .^ 2) / 2;
pixel2MicronMagnitude = sqrt(pixel2MicronArea);
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
    vImg = [1 1 1 1 1; ...
        1 1 1 1 1; ...
        1 1 1 1 1; ...
        1 1 1 1 1; ...
        1 1 1 1 1];
    [vPattern(:,1) vPattern(:,2)] = find(vImg == 1);
    vPattern(:,1) = vPattern(:,1) - ceil(size(vImg, 1) / 2);
    vPattern(:,2) = vPattern(:,2) - ceil(size(vImg, 2) / 2);
    
    % Construct the values for the contour and skeleton curvature heat map.
    blue = zeros(360, 3);
    blue(:,3) = 1;
    red = zeros(360, 3);
    red(:,1) = 1;
    cRGB = [blue(1:90,:); jet(181); red(1:90,:)]; % thermal
    sRGB = [blue(1:90,:); jet(181); red(1:90,:)]; % thermal
    sRGBNaN = [1 0 0]; % red
    hRGB = [0 1 0]; % green
    vRGB = [1 0 0]; % red
end

% Pre-allocate memory.
vulvaContours(1:samples,1:2,1:length(worms)) = NaN;
nonVulvaContours(1:samples,1:2,1:length(worms)) = NaN;
skeletons(1:samples,1:2,1:length(worms)) = NaN;
angles(1:samples,1:length(worms)) = NaN;
inOutTouches(1:samples,1:length(worms)) = NaN;
lengths(1:length(worms)) = NaN;
widths(1:samples,1:length(worms)) = NaN;
headAreas(1:length(worms)) = NaN;
tailAreas(1:length(worms)) = NaN;
vulvaAreas(1:length(worms)) = NaN;
nonVulvaAreas(1:length(worms)) = NaN;
isNormed(1:length(worms)) = false;

% Downsample the worms and convert them to absolute coordinates.
j = 1;
for i = 1:length(worms)
    
    % Get the frame number.
    % Note: the video indices from 0 whereas Matlab indices from 1;
    % therefore, we offset the frame number by 1.
    worm = worms{i};
    if isstruct(worm)
        frame = worm.video.frame + 1;
    elseif iscell(worm)
        frame = worm{1}{1} + 1;
        
    % Segmentation failed.
    else
        continue;
    end
    
    % Where is the stage?
    while j < size(moves, 1) && frame >= moves(j + 1,1)
        j = j + 1;
    end
    
    % The stage is moving.
    if frame >= moves(j,1) && frame <= moves(j,2)
%         vulvaContours(:,:,i) = NaN;
%         nonVulvaContours(:,:,i) = NaN;
%         skeletons(:,:,i) = NaN;
%         angles(:,i) = NaN;
%         inOutTouches(:,i) = NaN;
%         lengths(i) = NaN;
%         widths(:,i) = NaN;
%         areas(i) = NaN;
%         headAreas(i) = NaN;
%         tailAreas(i) = NaN;
%         vulvaAreas(i) = NaN;
%         nonVulvaAreas(i) = NaN;
        continue;
    end
    
    % The worm information is in a struct.
    if isstruct(worm)
        
        % Extract the contour information.
        cPixels = worm.contour.pixels;
        cHeadI = worm.contour.headI;
        cTailI = worm.contour.tailI;
        cCCLengths = worm.contour.chainCodeLengths;
        
        % Extract the skeleton information.
        sPixels = worm.skeleton.pixels;
        sAngles = worm.skeleton.angles;
        sCCLengths = worm.skeleton.chainCodeLengths;
        sWidths = worm.skeleton.widths;
        
        % Extract the area information.
        hArea = worm.head.area;
        tArea = worm.tail.area;
        lArea = worm.left.area;
        rArea = worm.right.area;
        
        % Extract the orientation information.
        isHeadFlipped = worm.orientation.head.isFlipped;
        isVulvaClockwiseFromHead = ...
            worm.orientation.vulva.isClockwiseFromHead;
        
    % The worm information is in a cell.
    else
        
        % Extract the contour information.
        cPixels = worm{2}{1};
        cHeadI = worm{2}{6};
        cTailI = worm{2}{7};
        cCCLengths = worm{2}{8};
        
        % Extract the skeleton information.
        sPixels = worm{3}{1};
        sAngles = worm{3}{6};
        sCCLengths = worm{3}{8};
        sWidths = worm{3}{9};
        
        % Extract the area information.
        hArea = worm{4}{3};
        tArea = worm{5}{3};
        lArea = worm{6}{3};
        rArea = worm{7}{3};
        
        % Extract the orientation information.
        isHeadFlipped = worm{8}{1}{1};
        isVulvaClockwiseFromHead = worm{8}{2}{1};
    end
    
    % Split the contour into sides.
    % Side1 always goes from head to tail in positive, index increments.
    % Side2 always goes from head to tail in negative, index increments.
    if cHeadI <= cTailI
        side1 = (cHeadI:cTailI)';
        ccl1 = cCCLengths(cHeadI:cTailI) - cCCLengths(cHeadI);
        side2 = [cTailI:size(cPixels,1), 1:cHeadI]';
        ccl2 = [cCCLengths(cTailI:end) - cCCLengths(cTailI); ...
            cCCLengths(1:cHeadI) + cCCLengths(end) - cCCLengths(cTailI)];
    else % cHeadI > cTailI
        side1 = [cHeadI:size(cPixels,1), 1:cTailI]';
        ccl1 = [cCCLengths(cHeadI:end) - cCCLengths(cHeadI); ...
            cCCLengths(1:cTailI) + cCCLengths(end) - cCCLengths(cHeadI)];
        side2 = (cTailI:cHeadI)';
        ccl2 = cCCLengths(cTailI:cHeadI) - cCCLengths(cTailI);
    end

    % Compute the clockwise and anti-clockwise sides from the head.
    if isHeadFlipped
        clockSide = side2;
        clockCCL = ccl2;
        antiSide = flipud(side1);
        antiCCL = flipud(ccl1(end) - ccl1);
    else
        clockSide = side1;
        clockCCL = ccl1;
        antiSide = flipud(side2);
        antiCCL = flipud(ccl2(end) - ccl2);
    end
        
    % Compute the vulval and non-vulval sides' contours.
    if isVulvaClockwiseFromHead
        vIndices = clockSide;
        vCCLengths = clockCCL;
        nvIndices = antiSide;
        nvCCLengths = antiCCL;
    else
        vIndices = antiSide;
        vCCLengths = antiCCL;
        nvIndices = clockSide;
        nvCCLengths = clockCCL;
    end
    
    % Are the contours and skeleton long enough?
    if length(vIndices) < samples
        warning('normWorms:VulvaContourTooShort', ['The ventral ' ...
            'contour is shorter than the sampling points requested']);
        continue;
    end
    if length(nvIndices) < samples
        warning('normWorms:NonVulvaContourTooShort', ['The dorsal ' ...
            'contour is shorter than the sampling points requested']);
        continue;
    end
    if size(sPixels, 1) < samples
        warning('normWorms:SkeletonTooShort', ['The skeleton is ' ...
            'shorter than the sampling points requested']);
        continue;
    end
    
    % Downsample the contours.
    normVPixels = downSamplePoints(cPixels(vIndices,:), samples, ...
        vCCLengths);
    normNVPixels = downSamplePoints(cPixels(nvIndices,:), samples, ...
        nvCCLengths);
    
    % Convert the contours to absolute coordinates.
    vulvaContours(:,:,i) = pixels2Microns(origins(j,:), ...
        fliplr(normVPixels), pixel2MicronScale, rotation);
    nonVulvaContours(:,:,i) = pixels2Microns(origins(j,:), ...
        fliplr(normNVPixels), pixel2MicronScale, rotation);
    
    % Downsample the skeleton and convert it to absolute coordinates.
    [normSPixels normSIndices normSLengths] = ...
        downSamplePoints(sPixels, samples, sCCLengths);
    skeletons(:,:,i) = pixels2Microns(origins(j,:), ...
        fliplr(normSPixels), pixel2MicronScale, rotation);
    lengths(i) = sCCLengths(end) * pixel2MicronMagnitude;
    if isHeadFlipped
        skeletons(:,:,i) = flipud(skeletons(:,:,i));
        normSLengths = flipud(normSLengths);
        normSIndices = flipud(normSIndices);
    end
    
    % Downsample the skeleton angles.
    % Note: the lengths represent the head-to-tail order.
    angles(:,i) = chainCodeLengthInterp(sAngles, normSLengths, ...
            sCCLengths, normSIndices);
    if isVulvaClockwiseFromHead == isHeadFlipped
        angles(:,i) = -angles(:,i);
    end
    
    % Downsample the widths and convert them to absolute coordinates.
    % Note: the lengths represent the head-to-tail order.
    widths(:,i) = chainCodeLengthInterp(sWidths, normSLengths, ...
        sCCLengths, normSIndices) * pixel2MicronMagnitude;
    
    % Convert the areas to absolute coordinates.
    if isHeadFlipped
        tailAreas(i) = hArea * pixel2MicronArea;
        headAreas(i) = tArea * pixel2MicronArea;
    else
        headAreas(i) = hArea * pixel2MicronArea;
        tailAreas(i) = tArea * pixel2MicronArea;
    end
    if isVulvaClockwiseFromHead == isHeadFlipped
        vulvaAreas(i) = lArea * pixel2MicronArea;
        nonVulvaAreas(i) = rArea * pixel2MicronArea;
    else
        vulvaAreas(i) = rArea * pixel2MicronArea;
        nonVulvaAreas(i) = lArea * pixel2MicronArea;
    end
    
    % The worm was normalized.
    isNormed(i) = true;
    
    % Show the results in a figure.
    if verbose
        
        % Convert the worm to a structure.
        if ~isstruct(worm)
            worm = cell2worm(worm);
        end

        % Construct the normalized worm.
        nWorm = norm2Worm(frame, vulvaContours(:,:,i), ...
            nonVulvaContours(:,:,i), skeletons(:,:,i), angles(:,i), ...
            inOutTouches(:,i), lengths(i), widths(:,i), headAreas(i), ...
            tailAreas(i), vulvaAreas(i), nonVulvaAreas(i), ...
            origins(j,:), pixel2MicronScale, rotation, worm);
        
        % Determine the worm's MER (minimum enclosing rectangle).
        % Note: the skeleton can exit the contour.
        wMinX = min(min(worm.contour.pixels(:,2)), ...
            min(worm.skeleton.pixels(:,2)));
        wMaxX = max(max(worm.contour.pixels(:,2)), ...
            max(worm.skeleton.pixels(:,2)));
        wMinY = min(min(worm.contour.pixels(:,1)), ...
            min(worm.skeleton.pixels(:,1)));
        wMaxY = max(max(worm.contour.pixels(:,1)), ...
            max(worm.skeleton.pixels(:,1)));
        wMinX = min([wMinX, min(nWorm.contour.pixels(:,2)), ...
            min(nWorm.skeleton.pixels(:,2))]);
        wMaxX = max([wMaxX, max(nWorm.contour.pixels(:,2)), ...
            max(nWorm.skeleton.pixels(:,2))]);
        wMinY = min([wMinY, min(nWorm.contour.pixels(:,1)), ...
            min(nWorm.skeleton.pixels(:,1))]);
        wMaxY = max([wMaxY, max(nWorm.contour.pixels(:,1)), ...
            max(nWorm.skeleton.pixels(:,1))]);
        
        % Minimize the original worm.
        worm.contour.pixels(:,1) = worm.contour.pixels(:,1) - wMinY + 3;
        worm.contour.pixels(:,2) = worm.contour.pixels(:,2) - wMinX + 3;
        worm.skeleton.pixels(:,1) = worm.skeleton.pixels(:,1) - wMinY + 3;
        worm.skeleton.pixels(:,2) = worm.skeleton.pixels(:,2) - wMinX + 3;
        worm.head.pixels(:,1) = worm.head.pixels(:,1) - wMinY + 3;
        worm.head.pixels(:,2) = worm.head.pixels(:,2) - wMinX + 3;
        worm.tail.pixels(:,1) = worm.tail.pixels(:,1) - wMinY + 3;
        worm.tail.pixels(:,2) = worm.tail.pixels(:,2) - wMinX + 3;
        worm.left.pixels(:,1) = worm.left.pixels(:,1) - wMinY + 3;
        worm.left.pixels(:,2) = worm.left.pixels(:,2) - wMinX + 3;
        worm.right.pixels(:,1) = worm.right.pixels(:,1) - wMinY + 3;
        worm.right.pixels(:,2) = worm.right.pixels(:,2) - wMinX + 3;
        
        % Minimize the normalized worm.
        nWorm.contour.pixels(:,1) = nWorm.contour.pixels(:,1) - wMinY + 3;
        nWorm.contour.pixels(:,2) = nWorm.contour.pixels(:,2) - wMinX + 3;
        nWorm.skeleton.pixels(:,1) = ...
            nWorm.skeleton.pixels(:,1) - wMinY + 3;
        nWorm.skeleton.pixels(:,2) = ...
            nWorm.skeleton.pixels(:,2) - wMinX + 3;
        nWorm.head.pixels(:,1) = nWorm.head.pixels(:,1) - wMinY + 3;
        nWorm.head.pixels(:,2) = nWorm.head.pixels(:,2) - wMinX + 3;
        nWorm.tail.pixels(:,1) = nWorm.tail.pixels(:,1) - wMinY + 3;
        nWorm.tail.pixels(:,2) = nWorm.tail.pixels(:,2) - wMinX + 3;
        nWorm.left.pixels(:,1) = nWorm.left.pixels(:,1) - wMinY + 3;
        nWorm.left.pixels(:,2) = nWorm.left.pixels(:,2) - wMinX + 3;
        nWorm.right.pixels(:,1) = nWorm.right.pixels(:,1) - wMinY + 3;
        nWorm.right.pixels(:,2) = nWorm.right.pixels(:,2) - wMinX + 3;
        
        % Construct the worm images.
        emptyImg = zeros(wMaxY - wMinY + 5, wMaxX - wMinX + 5);
        oImg = overlayWormAngles(emptyImg, worm, cRGB, sRGB, sRGBNaN, ...
            hPattern, hRGB, 1, vPattern, vRGB, 1);
        nImg = overlayWormAngles(emptyImg, nWorm, cRGB, sRGB, sRGBNaN, ...
            hPattern, hRGB, 1, vPattern, vRGB, 1);
        
        % Show the worms.
        figure;
        subplot(1,2,1), imshow(nImg);
        title(['Normalized Worm (' num2str(samples) ' samples)']);
        subplot(1,2,2), imshow(oImg);
        title(['Original Worm (frame ' num2str(frame - 1) ')']);
    end
end
end
