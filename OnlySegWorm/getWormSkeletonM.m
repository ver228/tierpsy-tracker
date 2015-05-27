function [worm_results, worm] = getWormSkeletonM(maskData, frame, worm_results_prev, resampleNum)
%maskData = worm_mask; frame = current_frame; worm_results_prev = prev_worms{worm_index}; resampleNum = RESAMPLE_SIZE;

worm_results = [];
worm = segWormBWimgSimpleM(maskData, frame, 0.1, false);

if ~isempty(worm)
    % Orient and downsample the worm.
    tailI = worm.contour.tailI;
    headI = worm.contour.headI;
    contour_dorsal = worm.contour.pixels(headI:tailI,:);
    contour_ventral = [worm.contour.pixels(tailI:end,:); worm.contour.pixels(1:headI,:)];
    contour_ventral = flipud(contour_ventral);
    skeleton = worm.skeleton.pixels;
    cWidth = worm.skeleton.widths;
   
    %interpolate skeleton and contour. Additionally, curvspaceMex returns
    %the each curve length
    [worm_results.skeleton, worm_results.skeleton_length] = ...
        curvspaceMex(skeleton, resampleNum);
    [worm_results.contour_ventral, worm_results.contour_ventral_length] = ...
        curvspaceMex(contour_ventral, resampleNum);
    [worm_results.contour_dorsal, worm_results.contour_dorsal_length] = ...
        curvspaceMex(contour_dorsal, resampleNum);
    worm_results.frame = frame;
    
    %interpolate width
    xx = linspace(1, length(cWidth),resampleNum)';
    worm_results.width = interp1(cWidth, xx);
    %{
    midpoint = round(resampleNum/2);
    %shift coord point from the middle previous worm to the middle of the current worm
    %this should help with the alignment
    del_ske = worm_results.skeleton(midpoint,:)-worm_results_prev.skeleton(midpoint,:);
    for ff = {'skeleton', 'contour_ventral', 'contour_dorsal'}
        for nn = 1:2
            worm_results_prev.(ff{1})(:,nn) = worm_results_prev.(ff{1})(:,nn) + del_ske(nn);
        end
    end
    assert(all(abs(worm_results.skeleton(midpoint,:)-worm_results_prev.skeleton(midpoint,:))<1e-5));
    %}
    
    if ~isempty(worm_results_prev)
        %orient head and tail
        %check the the three points vs the last points to determine if the head
        %was switched compared with the prev_worm
        delta_head = sum(sum((worm_results_prev.skeleton(1:3,:) - worm_results.skeleton(1:3,:)).^2));
        delta_tail = sum(sum((worm_results_prev.skeleton(1:3,:) - worm_results.skeleton(end-2:end,:)).^2));
        assert(numel(delta_head) == 1);
        assert(numel(delta_tail) == 1);
        
        if delta_head > delta_tail
            worm_results.contour_dorsal = flipud(worm_results.contour_dorsal);
            worm_results.contour_ventral = flipud(worm_results.contour_ventral);
            worm_results.skeleton = flipud(worm_results.skeleton);
            worm_results.width = flipud(worm_results.width);
        end
    end
    
    %make sure the contours are clockwise direction
    worm_contour = [worm_results.contour_dorsal; worm_results.contour_ventral(end:-1:1,:)];
    signedArea = sum(worm_contour(1:end-1,1).*worm_contour(2:end,2)-worm_contour(2:end,1).*worm_contour(1:end-1,2));
    %x1y2 - x2y1(http://mathworld.wolfram.com/PolygonArea.html)
    if signedArea < 0
        dum = worm_results.contour_dorsal;
        worm_results.contour_dorsal = worm_results.contour_ventral;
        worm_results.contour_ventral = dum;
        
        dum = worm_results.contour_ventral_length;
        worm_results.contour_ventral_length = worm_results.contour_dorsal_length;
        worm_results.contour_dorsal_length = dum;
    end
    
    %remember to switch the head first
    %compare three pixels in the middle of the worm
    %delta_DD = sum((worm_results_prev.contour_dorsal(midpoint+(-1:1),:) - worm_results.contour_dorsal(midpoint+(-1:1),:)).^2);
    %delta_DV = sum((worm_results_prev.contour_dorsal(midpoint+(-1:1),:) - worm_results.contour_ventral(midpoint+(-1:1),:)).^2);
    
    %{
    %check overlap with the previous contour
    delta_DD = median(min(pdist2(worm_results_prev.contour_dorsal, worm_results.contour_dorsal),[],1));
    delta_DV = median(min(pdist2(worm_results_prev.contour_dorsal, worm_results.contour_ventral),[],1));
    
    assert(numel(delta_DD) == 1);
    assert(numel(delta_DV) == 1);
    
    if delta_DD > delta_DV
        dum = worm_results.contour_dorsal;
        worm_results.contour_dorsal = worm_results.contour_ventral;
        worm_results.contour_ventral = dum;
        
        dum = worm_results.contour_ventral_length;
        worm_results.contour_ventral_length = worm_results.contour_dorsal_length;
        worm_results.contour_dorsal_length = dum;
    end
    %}
    
end
