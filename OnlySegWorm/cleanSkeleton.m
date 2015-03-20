function [cSkeleton cWidths] = cleanSkeleton(skeleton, widths, wormSegSize)
%CLEANSKELETON Clean an 8-connected skeleton by removing any overlap and
%interpolating any missing points.
%
%   [CSKELETON] = CLEANSKELETON(SKELETON)
%
%   Note: the worm's skeleton is still rough. Therefore, index lengths, as
%         opposed to chain-code lengths, are used as the distance metric
%         over the worm's skeleton.
%
%   Input:
%       skeleton    - the 8-connected skeleton to clean
%       widths      - the worm's contour widths at each skeleton point
%       wormSegSize - the size (in contour points) of a worm segment.
%                     Note: the worm is roughly divided into 24 segments
%                     of musculature (i.e., hinges that represent degrees
%                     of freedom) on each side. Therefore, 48 segments
%                     around a 2-D contour.
%                     Note 2: "In C. elegans the 95 rhomboid-shaped body
%                     wall muscle cells are arranged as staggered pairs in
%                     four longitudinal bundles located in four quadrants.
%                     Three of these bundles (DL, DR, VR) contain 24 cells
%                     each, whereas VL bundle contains 23 cells." -
%                     www.wormatlas.org
%
%   Output:
%       cSkeleton - the cleaned skeleton (no overlap & no missing points)
%       cWidths   - the cleaned contour widths at each skeleton point
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% If a worm touches itself, the cuticle prevents the worm from folding and
% touching adjacent pairs of muscle segments; therefore, the distance
% between touching segments must be, at least, the length of 2 muscle
% segments.
maxSkeletonOverlap = 2 * wormSegSize;

% Remove small loops.
keep = 1:size(skeleton, 1); % points to keep
[~, pSort] = sortrows(skeleton); % the sorted points
[~, iSort] = sort(pSort); % index -> sorted point index
s1I = 1; % the first index for the skeleton loop
while s1I < length(pSort)
    
    % Find small loops.
    % Note: distal, looped sections are most likely touching;
    % therefore, we don't remove these.
    if ~isnan(keep(s1I))
        minI = s1I; % the minimum index for the loop
        maxI = s1I; % the maximum index for the loop
        
        % Search backwards.
        if iSort(s1I) > 1
            pI = iSort(s1I) - 1; % the index for the sorted points
            s2I = pSort(pI); % the second index for the skeleton loop
            dSkeleton = abs(skeleton(s1I,:) - skeleton(s2I,:));
            while any(dSkeleton <= 1)
                if s2I > s1I && ~isnan(keep(s2I)) && ...
                        all(dSkeleton <= 1) && ...
                        abs(s1I - s2I) < maxSkeletonOverlap
                    minI = min(minI, s2I);
                    maxI = max(maxI, s2I);
                end
                
                % Advance the second index for the skeleton loop.
                pI = pI - 1;
                if pI < 1
                    break;
                end
                s2I = pSort(pI);
                dSkeleton = abs(skeleton(s1I,:) - skeleton(s2I,:));
            end
        end
        
        % Search forwards.
        if  iSort(s1I) < length(pSort)
            pI = iSort(s1I) + 1; % the index for the sorted points
            s2I = pSort(pI); % the second index for the skeleton loop
            dSkeleton = abs(skeleton(s1I,:) - skeleton(s2I,:));
            while any(dSkeleton <= 1)
                if s2I > s1I && ~isnan(keep(s2I)) && ...
                        all(dSkeleton <= 1) && ...
                        abs(s1I - s2I) < maxSkeletonOverlap
                    minI = min(minI, s2I);
                    maxI = max(maxI, s2I);
                end
                
                % Advance the second index for the skeleton loop.
                pI = pI + 1;
                if pI > length(pSort)
                    break;
                end
                s2I = pSort(pI);
                dSkeleton = abs(skeleton(s1I,:) - skeleton(s2I,:));
            end
        end
        
        % Remove small loops.
        if minI < maxI
            
            % Remove the overlap.
            if isequal(skeleton(minI,:), skeleton(maxI,:))
                keep((minI + 1):maxI) = nan;
                widths(minI) = min(widths(minI:maxI));
                
            % Remove the loop.
            elseif minI < maxI - 1
                keep((minI + 1):(maxI - 1)) = nan;
                widths(minI) = min(widths(minI:(maxI - 1)));
                widths(maxI) = min(widths((minI + 1):(maxI)));
            end
            
        end
        
        % Advance the first index for the skeleton loop.
        if s1I < maxI
            s1I = maxI;
        else
            s1I = s1I + 1;
        end
        
    % Advance the first index for the skeleton loop.
    else
        s1I = s1I + 1;
    end
end
skeleton = skeleton(~isnan(keep),:);
widths = widths(~isnan(keep));

% The head and tail have no width.
widths(1) = 0;
widths(end) = 0;

% Heal the skeleton by interpolating missing points.
cSkeleton = zeros(2 * size(skeleton, 1), 2); % pre-allocate memory
cWidths = zeros(2 * size(skeleton, 1), 1); % pre-allocate memory
j = 1;
for i = 1:(length(skeleton) - 1)
    
    % Initialize the point differences.
    y = abs(skeleton(i + 1,1) - skeleton(i,1));
    x = abs(skeleton(i + 1,2) - skeleton(i,2));
    
    % Add the point.
    if (y == 0 || y == 1) && (x == 0 || x == 1)
        cSkeleton(j,:) = skeleton(i,:);
        cWidths(j) = widths(i);
        j = j + 1;
        
    % Interpolate the missing points.
    else
        points = max(y, x);
        y1 = skeleton(i,1);
        y2 = skeleton(i + 1,1);
        x1 = skeleton(i,2);
        x2 = skeleton(i + 1,2);
        cSkeleton(j:(j + points),1) = round(linspace(y1, y2, points + 1));
        cSkeleton(j:(j + points),2) = round(linspace(x1, x2, points + 1));
        cWidths(j:(j + points)) = round(linspace(widths(i), ...
            widths(i + 1), points + 1));
        j = j + points;
    end
end

% Add the last point.
if (cSkeleton(1,1) ~= skeleton(end,1)) || ...
        (cSkeleton(1,2) ~= skeleton(end,2))
    cSkeleton(j,:) = skeleton(end,:);
    cWidths(j) = widths(end);
    j = j + 1;
end

% Collapse any extra memory.
cSkeleton(j:end,:) = [];
cWidths(j:end) = [];

% Anti alias.
keep = 1:size(cSkeleton, 1); % points to keep
i = 1;
endI = size(cSkeleton, 1) - 1;
while i < endI
    
    % Smooth any stairs.
    nextI = i + 2;
    if abs(cSkeleton(i,1) - cSkeleton(nextI,1)) <= 1 && ...
            abs(cSkeleton(i,2) - cSkeleton(nextI,2)) <= 1
        keep(i + 1) = nan;
        
        % Advance.
        i = nextI;
        
    % Advance.
    else
        i = i + 1;
    end
end
cSkeleton = cSkeleton(~isnan(keep),:);
cWidths = cWidths(~isnan(keep));
end
