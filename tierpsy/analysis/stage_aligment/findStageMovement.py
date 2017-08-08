import numpy as np
import warnings
import tables
from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name
from tierpsy.helper.params import read_fps
from tierpsy.analysis.stage_aligment.get_mask_diff_var import get_mask_diff_var

def _matlab_std(x):    
    if x.size <= 1:
        #in array of size 1 MATLAB returns 0 in the std while numpy nan
        return 0.
    else:
        #ddof=1 to have the same behaviour as MATLAB
        return np.nanstd(x, ddof=1)
    
    

def getFrameDiffVar(masked_file, progress_refresh_rate_s=100):
    base_name = get_base_name(masked_file)
    progress_prefix = '{} Calculating variance of the difference between frames.'.format(base_name)
    
    with tables.File(masked_file, 'r') as fid:
        masks = fid.get_node('/mask')

        tot, w, h = masks.shape
        progress_time = TimeCounter(progress_prefix, tot)
        fps = read_fps(masked_file, dflt=25)
        progress_refresh_rate = int(round(fps*progress_refresh_rate_s))

        img_var_diff = np.zeros(tot-1)
        frame_prev = masks[0]
        for ii in range(1, tot):
            frame_current = masks[ii]
            img_var_diff[ii-1] = get_mask_diff_var(frame_current, frame_prev)
            frame_prev = frame_current;

            if ii % progress_refresh_rate == 0:
                print_flush(progress_time.get_str(ii))

        if tot>1:
            print_flush(progress_time.get_str(ii))
    return img_var_diff

def graythreshmat(I_ori):
    #reimplementation of the matlab graythresh for consistency
    
    #it convert the image into a uint8 if it is a double it assumes 
    #it is between 0 and 1, 
    
    I = I_ori.copy()
    
    #make nan zeros (that's what matlab does)
    I[np.isnan(I)]=0
    I[I<0]=0
    assert np.all(I>=0) and np.all(I<=1)
    
    I = np.round(I*255).astype(np.uint8)
    
    if np.all(I == I[0]):
        #if all values are equal return 0
        return 0
    
    num_bins = 256;
    counts = np.bincount(I, minlength=num_bins);
    p = counts/np.sum(counts)
    omega = np.cumsum(p)
    mu = np.cumsum(p *(np.arange(1, num_bins+1)));
    
    mu_t = mu[-1]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sigma_b_squared = ((mu_t * omega - mu)**2) / (omega * (1 - omega));
    
    
    if not np.all(np.isnan(sigma_b_squared)):
        maxval = np.nanmax(sigma_b_squared);
        idx = np.mean(np.where(sigma_b_squared == maxval)[0]);
        level = idx / (num_bins - 1);
    else:
        level = 0
        
    return level

def _get_small_otsu(frame_diffs, th):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        small_diffs = frame_diffs[frame_diffs < th];
        small_th = np.nanmedian(small_diffs) + 3 * _matlab_std(small_diffs);
    return small_diffs, small_th



def maxPeaksDistHeight(x, dist, height):
    """
    %MAXPEAKSDISTHEIGHT Find the maximum peaks in a vector. The peaks are
    %   separated by, at least, the given distance unless interrupted and are, at least, the given
    %   height.
    %
    %   [PEAKS INDICES] = MAXPEAKSDISTHEIGHT(X, DIST, HEIGHT)
    %
    %   Input:
    %       x      - the vector to search for maximum peaks
    %       dist   - the minimum distance between peaks
    %       height - the minimum height for peaks
    %
    %   Output:
    %       peaks   - the maximum peaks
    %       indices - the indices for the peaks
    %
    %
    % © Medical Research Council 2012
    % You will not remove any copyright or other notices from the Software; 
    % you must reproduce all copyright notices and other proprietary 
    % notices on any copies of the Software.
    """

    #% Is the vector larger than the search window?
    winSize = 2 * dist + 1;
    if x.size < winSize:
        peak = np.nanmax(x)
        if peak < height:
            return np.zeros(0), np.zeros(0)
    
    #initialize variables
    peaks = []
    indices = []
    im = None; #% the last maxima index
    ip = None; #% the current, potential, max peak index
    p = None; #% the current, potential, max peak value
    i = 0; #% the vector index
    
    #% Search for peaks.
    while i < x.size:
        #% Found a potential peak.
        if (x[i] >= height) and ((p is None) or (x[i] > p)):
            ip = i;
            p = x[i];
        
        
        #% Test the potential peak.
        if not p is None and (i - ip >= dist) or i == x.size-1:
            #% Check the untested values next to the previous maxima.
            if im is not None and ip - im <= 2 * dist:
                #% Record the peak.
                if p > np.nanmax(x[(ip - dist):(im + dist+1)]):
                    indices.append(ip);
                    peaks.append(p);
                
                #% Record the maxima.
                im = ip;
                ip = i;
                p = x[ip];
                
            #% Record the peak.
            else:
                indices.append(ip);
                peaks.append(p);
                im = ip;
                ip = i;
                p = x[ip];
            
        #% Advance.
        i = i + 1;
    
    return np.array(peaks), np.array(indices)
#%%
def _initial_checks(mediaTimes, locations, delayFrames, fps):
    if fps < 0.1 or fps > 100:
        warnings.warn('WeirdFPS: recorded at {} frames/second. An unusual frame rate'.format(fps))
    
    if mediaTimes.size > 0 and mediaTimes[0] != 0:
        raise ValueError('NoInitialMediaTime. The first media time must be 0')
    
    
    if not isinstance(delayFrames, int):
        delayFrames = int(delayFrames)
    #%%

    # Save the spare 0 media time location in case the corresponding
    # stage-movement, frame difference occured after the video started.
    spareZeroTimeLocation = [];
    # If there's more than one initial media time, use the latest one.
    if (mediaTimes.size > 1):
        i = 1;
        while (i < mediaTimes.size and mediaTimes[i] == 0):
            i = i + 1;
        
        
        if i > 1:
            spareZeroTimeLocation = locations[i - 2,:];
        
            #% Dump the extraneous 0 media times and locations.
            mediaTimes = mediaTimes[i-1:]
            locations = locations[i-1:]
    #%%
    return mediaTimes, locations, delayFrames, fps, spareZeroTimeLocation 

def _norm_frame_diffs(frameDiffs): 
    
    #% Are there enough frames?
    if np.sum(~np.isnan(frameDiffs)) < 2:
        raise ValueError('InsufficientFrames. The video must have at least 2, non-dropped frames');
    
    #% No frame difference means the frame was dropped.
    frameDiffs[frameDiffs == 0] = np.nan;
    
    #% Normalize the frame differences and shift them over one to align them
    #% with the video frames.
    frameDiffs /= np.nanmax(frameDiffs)
    frameDiffs = np.insert(frameDiffs, 0 , frameDiffs[0])
    
    return frameDiffs
    
#%%

        
#%%
def _init_search(frameDiffs, gOtsuThr, gSmallDiffs, gSmallThr, 
                 mediaTimes, maxMoveFrames, fps):    
    #%%
    #% The log file doesn't contain any stage movements.
    if mediaTimes.size < 2:
        warnings.warn('NoStageMovements. The stage never moves');
        
        #% Are there any large frame-difference peaks?
        if gOtsuThr >= gSmallThr:
            _, indices = maxPeaksDistHeight(frameDiffs, maxMoveFrames-1, gOtsuThr);
        
            warnings.warn('UnexpectedPeaks. There are {} large frame-difference ' \
                          'peaks even though the stage never moves'.format(indices.size));
        return None
        

    #% Does the Otsu threshold separate the 99% of the small frame differences
    #% from the large ones?
    if gSmallDiffs.size==0 or gOtsuThr < gSmallThr:
        warnings.warn("NoGlobalOtsuThreshold. Using the Otsu method, as a whole, " \
                     "the frame differences don't appear to contain any distinguishably " \
                     "large peaks (corresponding to stage movements). Trying half of the " \
                     "maximum frame difference instead.")
        
        #% Try half the maximum frame difference as a threshold to distinguish large peaks.
        gOtsuThr = 0.5
        gSmallDiffs, gSmallThr = _get_small_otsu(frameDiffs, gOtsuThr)
        
        #% Does a threshold at half the maximum frame difference separate the
        #% 99% of the small frame differences from the large ones?
        if gSmallDiffs.size==0 or gOtsuThr < gSmallThr:
            warnings.warn('NoGlobalThresholds.  Cannot find a global threshold to ' \
                          'distinguish the large frame-difference peaks.');
            gOtsuThr = np.nan;
            gSmallThr = np.nan;
    #%%
    #% Pre-allocate memory.

    print('S', mediaTimes.size)
    frames = np.zeros(frameDiffs.shape); #% stage movement status for frames
    movesI = np.full((mediaTimes.size, 2), -100, np.int)
    movesI[0,:] = 0;
    
    #% Compute the search boundary for the first frame-difference peak.
    maxMoveTime = maxMoveFrames / fps; #% maximum time a movement takes
    timeOff = maxMoveTime; #% the current media time offset
    peakI = 0; # the current stage movement peak's index
    prevPeakI = 0; # the previous stage-movement peak's index
    prevPeakEndI = 0; # the previous stage-movement peak's end index
    startI = 0; # the start index for our search
    
    endI = 2 * maxMoveFrames-1 #due to the different index from python and matlab
    endI = min(endI, frameDiffs.size-1); #% the end index for our search
    
    
    searchDiffs = frameDiffs[startI:endI+1];
    
    
    #% Is the Otsu threshold large enough?
    otsuThr = graythreshmat(searchDiffs);
    isOtsu = otsuThr > gOtsuThr; #% false if no global Otsu
    
    if not isOtsu:
        # Does the Otsu threshold separate the 99% of the small frame
        # differences from the large ones? And, if there is a global small
        # threshold, is the Otsu threshold larger?
        smallDiffs, smallThr = _get_small_otsu(searchDiffs, otsuThr)
        isOtsu = (smallDiffs.size > 0) & \
            np.any(~np.isnan(smallDiffs)) & \
            (np.isnan(gSmallThr) | (otsuThr > gSmallThr)) & \
            (otsuThr >= smallThr)
    
        # Does the global Otsu threshold pull out any peaks?
        if not isOtsu and \
        not np.isnan(gOtsuThr) and \
        np.any(searchDiffs > gOtsuThr):
            otsuThr = gOtsuThr;
            isOtsu = True;
    
    if isOtsu:
        #% Do the frame differences begin with a stage movement?
        indices, = np.where(searchDiffs > otsuThr);
        firstPeakI = indices[0];
        
        if firstPeakI < maxMoveFrames:
            #% Find the largest frame-difference peak.
            peakI = np.nanargmax(frameDiffs[:maxMoveFrames+1]);
            prevPeakI = peakI;
            #% Compute the media time offset.
            timeOff = (peakI +1) / fps;
            
        # Is there a still interval before the first stage movement?
        if peakI > 0:
            i = peakI - 1;
            while i > 0:
                if frameDiffs[i] < gSmallThr and frameDiffs[i - 1] < gSmallThr:
                    peakI = 0;
                    break
                i -= 1
    
    #% We reached the end.
    endI = peakI + maxMoveFrames;
    if endI >= frameDiffs.size-1:
        prevPeakEndI = frameDiffs.size;
        #% Find a temporary front end for a potential initial stage movement.
    else:
        searchDiffs = frameDiffs[peakI:endI+1];
        # Does the search window contain multiple stage movements?
        if not (np.isnan(gOtsuThr) or np.isnan(gSmallThr)):
            
            foundMove = False;
            for i in range(searchDiffs.size):
                #% We found a still interval.
                if not foundMove and searchDiffs[i] < gSmallThr:
                    foundMove = True;
                    
                # We found the end of the still interval, cut off the rest.
                elif foundMove and searchDiffs[i] > gSmallThr:
                    searchDiffs = searchDiffs[0:(i - 1)]
                    break
        
        # Find a temporary front end for a potential initial stage movement.
        i = np.nanargmin(searchDiffs);
        peakFrontEndI = peakI + i;
        minDiff = searchDiffs[i]
        
        
        # If the temporary front end's frame difference is small, try to push
        # the front end backwards (closer to the stage movement).
        if minDiff <= gSmallThr:
            i = peakI
            while i < peakFrontEndI:    
                if frameDiffs[i] <= gSmallThr:
                    peakFrontEndI = i;
                    break;
                i += 1
        
        #% If the temporary front end's frame difference is large, try to
        #% push the front end forwards (further from the stage movement).
        elif minDiff >= gOtsuThr or \
        (minDiff > gSmallThr and \
         peakFrontEndI < endI and \
         np.all(np.isnan(frameDiffs[(peakFrontEndI + 1):endI]))):
            peakFrontEndI = endI;
        
        
        prevPeakEndI = peakFrontEndI;
        #%%
    return frames, movesI, prevPeakI, prevPeakEndI, maxMoveTime, timeOff
#%%
def _get_search_diff(frameDiffs, prevPeakEndI, mediaTimeOffI, maxMoveFrames):
    startI = prevPeakEndI;
    # Compute the search boundary for matching frame-difference peaks.
    
    x1 = startI + 2 * abs(mediaTimeOffI - (startI+1))
    x2 = max((startI+1), mediaTimeOffI) + maxMoveFrames - 1
    endI = min(max(x1, x2), frameDiffs.size-1)
    
    searchDiffs = frameDiffs[startI:endI+1];
    return searchDiffs, startI, endI

#%%
def get_otsu_thresh(frameDiffs, 
                    searchDiffs, 
                    gOtsuThr, 
                    gSmallThr, 
                    prevOtsuThr, 
                    prevSmallThr):
    #% Is the Otsu threshold large enough?
    otsuThr = graythreshmat(searchDiffs);
    
    isOtsu = otsuThr > prevSmallThr or otsuThr > gOtsuThr;
    if not isOtsu:
        #% Does the Otsu threshold separate the 99% of the small frame
        #% differences from the large ones?
        if np.isnan(prevSmallThr) or otsuThr > prevSmallThr or otsuThr > gSmallThr:
            smallDiffs, smallThr = _get_small_otsu(frameDiffs, otsuThr)
            isOtsu = (len(smallDiffs)>0) & np.any(~np.isnan(smallDiffs)) & (otsuThr >= smallThr);
        
        #% Try the global Otsu threshold or, if there is none, attempt to
        #% use half the search window's maximum frame difference.
        if not isOtsu:
            #% Try using half the search window's maximum frame difference.
            if np.isnan(gOtsuThr):
                otsuThr = np.nanmax(searchDiffs) / 2;
                
                #% Does the half-maximum threshold separate the 99% of the
                #% small frame differences from the large ones?
                smallDiffs, smallThr = _get_small_otsu(frameDiffs, otsuThr)
                isOtsu = smallDiffs & np.any(~np.isnan(smallDiffs)) & (otsuThr >= smallThr);
                
            #% Does the global Otsu threshold pull out any peaks?
            elif np.any(searchDiffs > gOtsuThr):
                otsuThr = gOtsuThr;
                isOtsu = True;
                
            #% Does the global Otsu threshold pull out any peaks?
            elif np.any(searchDiffs > prevOtsuThr):
                otsuThr = prevOtsuThr;
                isOtsu = True;
    
    return isOtsu, otsuThr

    
def _get_peak_indices(frameDiffs, 
                      searchDiffs, 
                      isOtsu, 
                      otsuThr, 
                      gOtsuThr, 
                      gSmallThr,
                      prevOtsuThr, 
                      prevSmallThr,
                      maxMoveFrames):
    #% Match the media time stage movement to a peak.
   
    if not isOtsu:
        indices = [];
    else:
        #% Compute and set the global thresholds.
        if np.isnan(gOtsuThr):
            #% Use a small threshold at 99% of the small frame differences.
            smallDiffs, smallThr = _get_small_otsu(frameDiffs, gOtsuThr)
            #% Set the global thresholds.
            if otsuThr >= smallThr:
                gOtsuThr = otsuThr;
                gSmallThr = smallThr;
                
                #% Set the previous small threshold.
                if np.isnan(prevOtsuThr):
                    prevOtsuThr = otsuThr;
                    prevSmallThr = smallThr;
                
                
            #% Use the previous small threshold.
            elif not np.isnan(prevSmallThr):
                smallThr = prevSmallThr;   
        #% Compute the local thresholds.
        else:
            otsuThr = min(otsuThr, gOtsuThr);
            smallThr = max(prevSmallThr, gSmallThr);
            if smallThr > otsuThr:
                smallThr = min(prevSmallThr, gSmallThr);
        
        #% Does the search window contain multiple stage movements?
        foundMove = False;
        for j in range(searchDiffs.size):
            #% We found a stage movement.
            if not foundMove and searchDiffs[j] > otsuThr:
                foundMove = True;
                
            #% We found the end of the stage movement, cut off the rest.
            elif foundMove and searchDiffs[j] < smallThr:
                searchDiffs = searchDiffs[0:j];
                break;
        
        
        #% Find at least one distinguishably large peak.
        _, indices = maxPeaksDistHeight(searchDiffs, maxMoveFrames, otsuThr);
    return indices, prevOtsuThr, prevSmallThr


#%%
def findStageMovement(frameDiffs, mediaTimes, locations, delayFrames, fps):

    '''
    %MODIFIED FROM SEGWORM AEJ. This help is outdated, I'll modified later. AEJ
    
    %FINDSTAGEMOVEMENT Find stage movements in a worm experiment.
    %
    % The algorithm is as follows:
    %
    % 4. If there are no stage movements, we're done.
    %
    % 5. The log file sometimes contains more than one entry at 0 media time.
    % These represent stage movements that may have completed before the video
    % begins. Therefore, we don't use them but, instead, store them in case we
    % find their stage movement in the video frame differences.
    %
    % 6. The frame differences need to be aligned to the video frames.
    % Therefore, we copy the first one and shift the rest over so that the
    % frame differences are one-to-one with the video frames. Note that video
    % indexing starts at 0 while Matlab indexing begins at 1. Also note, due
    % to differentiation, large frame differences that occur at the beginning
    % of a stage movement now represent the first frame of movement. Large
    % frame differences that occur at the end of a stage movement now represent
    % the first non-movement frame.
    %
    % 7. Compute the global Otsu threshold for the frame-differences to
    % distinguish stage-movement peaks. Then compute a global non-movement
    % threshold by taking all values less than the Otsu, and computing 3
    % standard deviations from the median (approximately 99% of the small
    % values). Please note, stage movements ramp up/down to
    % accelerate/decelerate to/from the target speed. Therefore, the values
    % below the Otsu threshold are contaminated with stage movement
    % acceleration and decelaration. Fortunately, non-movement frames account
    % for more than 50% of the frame differences. Therefore, to avoid the stage
    % movement contamination, we use the median of the small values rather than
    % their mean when computing the global non-movement (small) threshold. If
    % this small threshold is greater than the Otsu, we've made a poor choice
    % and ignore both thresholds. Otherwise, these 2 global thresholds serve as
    % a backup to the local ones when distinguishing stage movements.
    %
    % 8. Ocasionally, computing the global Otsu threshold fails. This occurs
    % when a long video has few stage movements. In this case, stage movements
    % appear to be rare outliers and the Otsu method minimizes in-group
    % variance by splitting the non-stage movement modality into two groups
    % (most likely periods of worm activity and inactivity). Under these
    % circumstances we attempt to use a global threshold at half the maximum
    % frame-difference variance. As detailed above, we test this threshold to
    % see whether it is sufficiently larger than 99% of the smaller movements.
    %
    % 9. When searching for stage movements, we use the same algorithm as the
    % one above(see step 7), over a smaller well-defined, search window, to
    % determine the local Otsu threshold. The local small threshold is computed
    % slightly differently (we use the mean instead of the median -- see step
    % 12 for an explanation). If the local Otsu threshold fails (it's smaller
    % than 99% of the values below it and smaller than the global Otsu), we
    % attempt to use the global one to see if it pulls out a peak.
    %
    % 10. A stage movement peak is defined as the largest value that exceeds
    % the minimum of the global and local Otsu thresholds. To avoid a situation
    % in which 2 stage movements occur within the same search window, we scan
    % the window for the first value exceeding the Otsu threshold and, if any
    % subsequent values drop below the minimum of global and local small
    % thresholds, we cut off the remainder of the window and ignore it.
    %
    % 11. Once a stage-movement peak is identified, we search for a temporary
    % back and front end to the movement. The stage movement must complete
    % within one delay time window (see step 2). Therefore, we search for the
    % minimum values, within one delay time window, before and after the peak.
    % The locations of these minimum values are the temporary back and front
    % ends for the stage movement. If the either value is below the small
    % threshold, we may have overshot the movement and, therefore, attempt to
    % find a location closer to the peak. Similarly, if either value is greater
    % than the maximum of the global and local small thresholds and the
    % remaining values till the end of the window are NaNs or, if either value
    % is greater than the Otsu threshold, we may have undershot the movement
    % and, therefore, attempt to find a location further from the peak.
    %
    % 12. Using one stage movement's temporary front end and the subsequent
    % movement's temporary back end, we compute the small threshold. This
    % interval is assumed to have no stage motion and, therefore, represents
    % frame-difference variance from a non-movement interval. The local small
    % threshold is defined as 3 deviations from the mean of this non-movement
    % interval (99% confidence). With this small threshold, we start at the
    % first peak and search forward for its real front end. Similarly, we start
    % at the subsequent peak and search backward for its real back end.
    %
    % 13. Conservatively, the beginning and end of the video are treated as the
    % end and begining of stage movements, respectively. We search for a
    % temporary front end and a temporary back end, respectively, using the
    % global small and Otsu thresholds.
    %
    % 14. After the final, logged stage motion is found in the frame
    % differences, we look to see if there are any subsequent, extra peaks.
    % An Otsu threshold is computed, as detailed earlier, using the interval
    % spanning from the final stage movement's temporary front end till the
    % final frame difference. If this threshold is unsuitable, we use the
    % global Otsu threshold. If we find any extra peaks, the first peak's back
    % end is located and its frame as well as the remainder of the frame
    % differences (including any other extra peaks) are marked as a single
    % large stage movement that extends till the end of the video. This is
    % necessary since Worm Tracker users can terminate logging prior to
    % terminating the video (e.g., this may occur automatically if the worm is
    % lost).
    %
    % 15. To find a stage movement, we compute its offset media time. The first
    % stage movement is offset by the delay time (see step 2). Subsequent media
    % times are offset by the difference between the previous media time and
    % its stage-movement peak. Therefore, each stage movement provides the
    % offset for the next one. The longer the wait till the next stage
    % movement, the less accurate this offset becomes. Therefore, we search for
    % each stage movement using a window that begins at the last stage
    % movement's temporary front end and ends at the offset media time plus
    % this distance (a window with its center at the offset media time). If the
    % window is too small (the offset media time is too close to the temporary
    % front end of the previous stage movement), we extend its end to be the
    % offset media time plus the delay time.
    %
    % 16. If a stage movement peak is absent, we attempt to shift the media
    % times backward or forward, relative to the stage movement peaks,
    % depending on whether the current peak is closer to the next or previous
    % media time, respectively. When shifting the media times backward, we
    % assume the first stage movement occurred prior to video recording and,
    % therefore, throw away its media time and location. When shifting the
    % media times forward, we look for a spare 0 media time and location (see
    % step 5). If there are no spares, we swallow up all the frames prior to
    % the end of the first stage movement and label them as an unusable
    % movement that began prior to recording and bled into the beginning of the
    % video.
    % 
    % 17. If we find a stage-movement peak closer to the previous offset media
    % time than its own supposed offset media time, we assume we have a
    % misalignment and attempt to shift the media times forward relative to the
    % stage movement peaks. There are some restrictions to this check since
    % small-scale, frame jitter can misrepresent the reported media time.
    %
    % 18. The final logged stage motion may occur after the video ends and, as
    % a result, may have no representation in the frame-difference variance.
    % Therefore, for the last stage movement, we check our threshold (see step
    % 10) to ensure that it separates 99% of the smaller values and, thereby,
    % picks up stage movement rather than splitting the non-movement modality.
    %
    %
    %
    %   FUNCTION [FRAMES INDICES LOCATIONS] = ...
    %       FINDSTAGEMOVEMENT(INFOFILE, LOGFILE, DIFFFILE, VERBOSE)
    %
    %   FUNCTION [FRAMES INDICES LOCATIONS] = ...
    %       FINDSTAGEMOVEMENT(INFOFILE, LOGFILE, DIFFFILE, VERBOSE, GUIHANDLE)
    %
    %   Input:
    %       infoFile  - the XML file with the experiment information
    %       logFile   - the CSV file with the stage locations
    %       diffFile  - the MAT file with the video differentiation
    %       verbose   - verbose mode 1 shows the results in a figure
    %                   verbose mode 2 labels the stage movements in the figure
    %       guiHandle - the GUI handle to use when showing the results;
    %                   if empty, the results are shown in a new figure
    %
    %   Output:
    %       frames    - a vector of frame status
    %                   true  = the frame contains stage movement
    %                   false = the frame does NOT contain stage movement
    %                   NaN   = the original video frame was dropped
    %                   Note: video frames are indexed from 0, Matlab indexes
    %                   from 1, please adjust your calculations accordingly
    %       movesI    - a 2-D matrix with, respectively, the start and end
    %                   frame indices of stage movements
    %       locations - the location of the stage after each stage movement
    %
    % See also VIDEO2DIFF
    %
    %
    % © Medical Research Council 2012
    % You will not remove any copyright or other notices from the Software; 
    % you must reproduce all copyright notices and other proprietary 
    % notices on any copies of the Software.
    '''
    
    #%%
    mediaTimes, locations, delayFrames, fps, spareZeroTimeLocation  = \
    _initial_checks(mediaTimes, locations, delayFrames, fps)
    frameDiffs = _norm_frame_diffs(frameDiffs)
    
    
    # Compute the global Otsu and small frame-difference thresholds.
    # Note 1: we use the Otsu to locate frame-difference peaks corresponding to
    # stage movement.
    # Note 2: we use the small threshold to locate the boundaries between
    # frame differences corresponding to stage movement and frame differences
    # corresponding to a non-movement interval.
    gOtsuThr = graythreshmat(frameDiffs);
    gSmallDiffs, gSmallThr = _get_small_otsu(frameDiffs, gOtsuThr)
    
    maxMoveFrames = delayFrames + 1; #% maximum frames a movement takes
    
    var_init = \
     _init_search(frameDiffs, gOtsuThr, gSmallDiffs, gSmallThr, 
                 mediaTimes, maxMoveFrames, fps)
    
    if var_init is not None:
        frames, movesI, prevPeakI, prevPeakEndI, maxMoveTime, timeOff = var_init
    else:
        #return empty vectors if there was no movement
        if len(locations) == 0:
            locations = np.zeros((1,2));
        return np.zeros(frameDiffs.size, np.int), np.zeros((1,2), np.int), locations
    

    #% Match the media time-stage movements to the frame-difference peaks.
    mediaTimeOff = 0.; #% the offset media time
    prevOtsuThr = gOtsuThr; #% the previous small threshold
    prevSmallThr = gSmallThr; #% the previous small threshold
    isShifted = False; #% have we shifted the data to try another alignment?
    
    #%%
    #AEJ I am using a while instead of a for to be able to go back
    i = 0
    while i < mediaTimes.size-1:
        i += 1
        #%%
        # Compute the offset media time.
        prevMediaTimeOff = mediaTimeOff;
        mediaTimeOff = mediaTimes[i] + timeOff;
        mediaTimeOffI = int(round(mediaTimeOff * fps));
        
        inputs_args = (frameDiffs, prevPeakEndI, mediaTimeOffI, maxMoveFrames)
        searchDiffs, startI, endI = _get_search_diff(*inputs_args)
        
        inputs_args = (frameDiffs, searchDiffs, gOtsuThr, gSmallThr, prevOtsuThr, prevSmallThr)
        isOtsu, otsuThr = get_otsu_thresh(*inputs_args)
    
        #% If we're at the end, make sure we're using an appropriate threshold.
        if i == mediaTimes.size-1:
            #% Does the threshold separate the 99% of the small frame
            #% differences from the large ones?
            smallDiffs, smallThr = _get_small_otsu(searchDiffs, otsuThr)
            isOtsu = (smallDiffs.size>0) & np.any(~np.isnan(smallDiffs)) & (otsuThr >= smallThr);
            
        inputs_args = (frameDiffs, searchDiffs, isOtsu, otsuThr, 
                       gOtsuThr, gSmallThr, prevOtsuThr, prevSmallThr, maxMoveFrames)
        
        indices, prevOtsuThr, prevSmallThr = _get_peak_indices(*inputs_args)
        
        
        #%%
        #% We can't find any distinguishably large peaks.
        peakI = np.nan;
        if len(indices) == 0:
            #% Does the last stage movement occur after the video ends?
            if i == mediaTimes.size-1 and endI >= frameDiffs.size-1:
                #% Does the last offset media time occur before the video ends?
                if mediaTimeOff < (frameDiffs.size - 1) / fps:
                    dd = 'LastPeak ' \
                        'The search window for the last stage movement ({}) ' \
                        'at media time {:.3f} seconds (frame {} ) offset to {:.3} ' \
                        'seconds (frame {}) to the last frame {:.3} seconds ' \
                        '(frame {}), does not have any distinguishably large peaks. '\
                        'The peak probably occured after the video ended and, ' \
                        'therefore, the last stage movement will be ignored.'
                    dd = dd.format(i, 
                                mediaTimes[i], 
                                round(mediaTimes[i] * fps),
                                mediaTimeOff,
                                startI - 1,
                                (endI - 1) / fps,
                                endI - 1
                                )
                    warnings.warn(dd)
                        
                # Ignore the last stage movement.
                mediaTimes = mediaTimes[:-1]
                locations = locations[:-1]
                movesI = movesI[:-1]
                break;
            
            #% Report the warning.
            dd = 'NoPeaks ' \
                        'The search window for stage movement ({}) ' \
                        'at media time {:.3f} seconds (frame {} ) offset to {:.3} ' \
                        'seconds (frame {}) to the last frame {:.3} seconds ' \
                        '(frame {}), does not have any distinguishably large peaks.'
            dd = dd.format(i, 
                        mediaTimes[i], 
                        round(mediaTimes[i] * fps),
                        mediaTimeOff,
                        startI - 1,
                        (endI - 1) / fps,
                        endI - 1
                        )
            warnings.warn(dd)
            
            
        # Use the first peak.
        else:
            #%%
            peakI = indices[0] + startI
            
            #% Is the current offset media time further from the frame-
            #% difference stage movement than the previous offset media time?
            peakTime = peakI / fps;
            timeDiff = mediaTimeOff - peakTime;
            prevTimeDiff = prevMediaTimeOff - peakTime;
            #%%
            if (i > 1) and \
            ((abs(prevTimeDiff) > maxMoveTime) or \
             (abs(timeDiff) > maxMoveTime)) and \
            (mediaTimeOff > prevMediaTimeOff) and \
            (abs(timeDiff / prevTimeDiff) > 2):
                
                 #% Report the warning.
                dd = ['FarPeak',
                     'Stage movement ({})'.format(i),
                     '(at media time {:.3f} seconds)'.format(mediaTimes[i]),
                     'offset to {:.3} seconds,'.format(mediaTimeOff),
                     'has its frame-difference peak at {:.3} seconds (frame {}),'.format(peakTime, peakI - 1),
                     'an error of {:.3} seconds.'.format(timeDiff),
                     'The previous media time, offset to {:.3} seconds,'.format(prevMediaTimeOff),
                     'is closer with an error only {:.3} seconds'.format(prevTimeDiff),
                     '(less than half the current media time error). ',
                     'Therefore, we probably have either a false ',
                     'peak, a shifted misalignment, or an abnormally long delay.'
                     ]
                dd = ' '.join(dd)
                
                warnings.warn(dd)
                
                #% Ignore this wrong peak.
                peakI = np.nan;
        #%%
        #% Can we realign (shift) the stage movements and media times?
        if np.isnan(peakI):
            lastMoveTime = movesI[i - 1, 0] / fps;
            isShiftable = True;
            if isShifted:
                isShiftable = False;
                
            #% Shift the media times forward.
            elif i > 1 and \
            abs(mediaTimes[i - 2] - lastMoveTime) < abs(mediaTimes[i] - lastMoveTime):
                #% Would a time shift align the media times with the
                #% frame-difference stage movements?
                for j in range(1, i - 1):
                    #% Compute the error from the predicted time.
                    offset =  movesI[j,0] / fps - mediaTimes[j - 1];
                    predictedTime = mediaTimes[j] + offset;
                    moveTime =  movesI[j + 1,0] / fps;
                    timeDiff = abs(predictedTime - moveTime);
                    
                    #% Compute the interval between the media times.
                    mediaDiff = mediaTimes[j] - mediaTimes[j - 1];
                    
                    #% Is the error in the predicted time greater than
                    #% the interval between media times?
                    if timeDiff > mediaDiff:
                        isShiftable = False;
                        break;
                
                #% Time cannot be shifted due to misalignment between the media
                #% times and frame-difference stage movements.
                if not isShiftable:
                    dd = 'TimeShiftAlignment ' \
                        'Time cannot be shifted forward because the' \
                        ' frame-difference stage movement at {:.3}'\
                        ' seconds would have a'\
                        ' predicted time of {:.3}'\
                        ' seconds (an error of {:.3}'  \
                        ' seconds) whereas the interval between its media' \
                        ' time and the previous media time is only {:.3}' \
                        ' seconds and,' \
                        ' therefore, smaller than the error from shifting.'
                    
                    dd.format(moveTime,
                              predictedTime,
                              timeDiff,
                              mediaDiff
                            )
                    warnings.warn(dd);
                    
                #% Shift everything forward using the spare 0 media time location.
                elif len(spareZeroTimeLocation)>0:
                    mediaTimes = np.insert(mediaTimes, 0,0)
                    locations = np.vstack((spareZeroTimeLocation, locations))
                    movesI = np.vstack((movesI, np.zeros((1,2), np.int)))
                    timeOff = (prevPeakI+1) / fps - mediaTimes[i - 1];
                    
                    #% Redo the match.
                    i = i - 1;
                    
                    #% Warn about the time shift.
                    warnings.warn('TimeShiftForward : ' \
                        'Shifting the media times forward relative to the ' \
                        'frame-difference stage movements (using a spare ' \
                        'location at media time 0:0:0.000) in an attempt ' \
                        'to realign them');
                    
                #% Shift everything forward by assuming a missing 0 media time
                #% location and swallowing earlier frames into the the first
                #% stage movement.
                else:
                    
                    frames[:movesI[1,0]] = True;
                    movesI[:(i - 1),:] = movesI[1:i,:];
                    movesI[0,0] = 0;
                    
                    timeOff = (prevPeakI+1) / fps - mediaTimes[i - 1];
                    
                    #% Redo the match.
                    i = i - 2;
                    
                    #% Warn about the time shift.
                    warnings.warn('TimeShiftForward : ' \
                        'Shifting the media times forward relative to the ' \
                        'frame-difference stage movements (by swallowing ' \
                        'earlier frames into the first stage movement) in ' \
                        'an attempt to realign them');
            # Shift the media times backward.
            else:
                #% Would a time shift align the media times with the
                #% frame-difference stage movements?
                for j in range(2, i - 1):
                    #% Compute the error from the predicted time.
                    offset =  movesI[j - 1,0] / fps - mediaTimes[j];
                    predictedTime = mediaTimes[j + 1] + offset;
                    moveTime =  movesI[j,0] / fps;
                    timeDiff = np.abs(predictedTime - moveTime);
                    
                    #% Compute the interval between the media times.
                    mediaDiff = mediaTimes[j + 1] - mediaTimes[j];
                    
                    #% Is the error in the predicted time greater than the
                    #% interval between media times?
                    if timeDiff > mediaDiff:
                        isShiftable = False;
                        break;
                
                #% Time cannot be shifted due to misalignment between the media
                #% times and frame-difference stage movements.
                if not isShiftable:
                    dd = ['TimeShiftAlignment',
                          'Time cannot be shifted backward because the',
                          'frame-difference stage movement at {:.3} seconds'.format(moveTime),
                          'would have a predicted time of {:.3} seconds'.format(predictedTime),
                          'seconds (an error of {:.3} seconds)'.format(timeDiff),
                          'whereas the interval between its media',
                          'time and the previous one is only {:.3} seconds'.format(mediaDiff),
                          'and, therefore, smaller than the error from shifting'
                          ]
                    warnings.warn(' '.join(dd))
                #% Shift everything backward.
                else:
                    
                    
                    mediaTimes = mediaTimes[1:];
                    locations = locations[1:];
                    movesI = movesI[:-1];
                    timeOff = (prevPeakI+1) / fps - mediaTimes[i - 1];
                    #% Redo the match.
                    i = i - 1;
                    
                    #% Warn about the time shift.
                    dd = 'TimeShiftBackward : ' \
                        'Shifting the media times backward relative to ' \
                        'the frame-difference stage movements in an ' \
                        'attempt to realign them'\
                    
                    warnings.warn(dd);
            
            #% Record the shift and continue.
            if isShiftable:
                isShifted = True;
                continue;
                
            #% We cannot realign (shift) the stage movements and media times.
            else:
                
                #% Compute the stage movement sizes.
                movesI = movesI[:i,:]
                moveSizes = np.zeros((movesI.shape[0],1));
                for j in range(2, movesI.shape[0] - 1):
                    moveDiffs = frameDiffs[movesI[j,0]:movesI[j,1]];
                    moveSizes[j] = np.nansum(moveDiffs)
                
                #% Compute the statistics for stage movement sizes.
                meanMoveSize = np.nanmean(moveSizes[1:]);
                stdMoveSize = _matlab_std(moveSizes[1:]);
                smallMoveThr = meanMoveSize - 2.5 * stdMoveSize;
                largeMoveThr = meanMoveSize + 2.5 * stdMoveSize;
                
                #% Are any of the stage movements considerably small or large?
                for j in range(1, movesI.shape[0]-1):
                    if moveSizes[j] < smallMoveThr:
                        #% Is the stage movement small?
                        before_f = movesI[j,0] - 1
                        after_f = movesI[j,1] - 1
                        #% Report the warning.
                        dd = ['ShortMove',
                             'Stage movement {}'.format(j),
                             'at media time {:.3}'.format(mediaTimes[j]),
                             'seconds (frame {}),'.format(int(round(mediaTimes[j] * fps))),
                             'spanning from {:.3} seconds (frame {})'.format(before_f / fps, before_f),
                             'to {:.3} seconds (frame {}),'.format(after_f/fps, after_f),
                             'is considerably small'
                             ]
                    elif moveSizes[j] > largeMoveThr:
                        #% Is the stage movement large?
                        before_f = movesI[j,0] - 1
                        after_f = movesI[j,1] - 1
                        
                        dd = ['LongMove',
                         'Stage movement {}'.format(j),
                         'at media time {:.3}'.format(mediaTimes[j]),
                         'seconds (frame {}),'.format(int(round(mediaTimes[j] * fps))),
                         'spanning from {:.3} seconds (frame {})'.format(before_f / fps, before_f),
                         'to {:.3} seconds (frame {}),'.format(after_f/fps, after_f),
                         'is considerably large'
                         ]
                    
                #% Construct the report.
                msg = 'NoShift : We cannot find a matching peak nor shift the time ' \
                     'for stage movement {} at media time {:.3} seconds (frame {:.3}).' \
                     .format(i,
                            mediaTimes[i],
                            mediaTimes[i]*fps)
            
                raise(ValueError(msg));
        
        if np.isnan(peakI):
            continue
        
        #% Find a temporary back end for this stage movement.
        #% Note: this peak may serve as its own temporary back end.
        startI = max(peakI - maxMoveFrames, prevPeakEndI);
        dd = frameDiffs[startI:peakI+1][::-1]
        j = np.nanargmin(dd)
        minDiff = dd[j]
        
        peakBackEndI = peakI - j; #% we flipped to choose the last min
        j = peakI - 1;
        
        #% If the temporary back end's frame difference is small, try to push
        #% the back end forwards (closer to the stage movement).
        if minDiff <= prevSmallThr:
            while j > startI:
                if frameDiffs[j] <= prevSmallThr:
                    peakBackEndI = j;
                    break;
                j -= 1;
            
        #% If the temporary back end's frame difference is large, try to push
        #% the back end backwards (further from the stage movement).
        elif minDiff >= min(otsuThr, gOtsuThr) or \
            (minDiff > gSmallThr and peakBackEndI > startI and \
             np.all(np.isnan(frameDiffs[startI:(peakBackEndI - 1)]))):
            peakBackEndI = startI;
         
        #% Compute a threshold for stage movement.
        smallDiffs = frameDiffs[prevPeakEndI:peakBackEndI+1];
        smallThr = np.nanmean(smallDiffs) + 3*_matlab_std(smallDiffs);
        
        if np.isnan(smallThr):
            smallThr = prevSmallThr;
        
        #% Find the front end for the previous stage movement.
        #set the range using the previous peak as range
        j = prevPeakI;
        while j < peakI and \
        (np.isnan(frameDiffs[j]) or \
        frameDiffs[j] > smallThr) and \
        (np.isnan(frameDiffs[j + 1]) or \
         frameDiffs[j + 1] > smallThr):
            j = j + 1;
        
        movesI[i - 1, 1] = j 
        prevPeakEndI = j-1;
        
        #%%
        #% Mark the previous stage movement.
        if movesI[i - 1,0] < 1:
            frames[:movesI[i - 1,1]] = True;
        else:
            frames[movesI[i - 1,0]:movesI[i - 1,1]] = True;
        
        
        #% Find the back end for this stage movement.
        j = peakI;
        while j > prevPeakEndI and \
        (np.isnan(frameDiffs[j]) or frameDiffs[j] > smallThr):
            j -= 1;
        
        movesI[i, 0] = j + 1;
        
        #% Is the non-movement frame-differences threshold too large?
        if smallThr <= otsuThr and (np.isnan(gOtsuThr) or smallThr <= gOtsuThr):
            prevOtsuThr = otsuThr;
            prevSmallThr = smallThr;
        else:
            before_f = movesI[i - 1,1] - 1
            after_f = movesI[i - 1,0] - 1
            dd = ['LargeNonMovementThreshold',
                  'The non-movement window between stage movement {}'.format(i-1),
                  'and stage movement {}'.format(i),
                  'from {:.3}  (frame {})'.format(before_f / fps, before_f),
                  'to {:.3}  (frame {})'.format(before_f / fps, after_f),
                  'contains considerably large frame-difference variance'
                  ]
            warnings.warn(' '.join(dd))
        
        
        #% Compute the media time offset.
        timeOff = peakTime - mediaTimes[i];
        
        #% We reached the end.
        endI = peakI + maxMoveFrames;
        if endI >= frameDiffs.size:
            peakFrontEndI = frameDiffs.size-1;
            
        #% Find a temporary front end for this stage movement.
        else:
            dd = frameDiffs[peakI+1:endI+1]
            j = np.nanargmin(dd)
            minDiff = dd[j]
            peakFrontEndI = peakI + j + 1;
            
            #% If the temporary front end's frame difference is large, try to
            #% push the front end forwards (further from the stage movement).
            
            if (minDiff >= min(otsuThr, gOtsuThr)) or \
                    (minDiff > max(smallThr, gSmallThr) and \
                    (peakFrontEndI < endI) and \
                    np.all(np.isnan(frameDiffs[(peakFrontEndI + 1):endI+1]))):
                peakFrontEndI = endI
            
        #% Try to push the temporary front end backwards (closer to the stage
        #% movement).
        j = peakI + 1;
        while j < peakFrontEndI:
            if frameDiffs[j] <= smallThr:
                peakFrontEndI = j;
                break;
            
            j = j + 1;
        
        #% Advance.
        prevPeakI = peakI;
        prevPeakEndI = peakFrontEndI;
        
        
    
    #% Do the frame differences end with a stage movement?
    if prevPeakEndI > frameDiffs.size:
        movesI[-1, 1] = frameDiffs.size;
        frames[movesI[-1,0]:] = True;
        movesI = np.vstack(movesI, np.full((1,2), frameDiffs.size+1))
        
    #% Find the front end for the last stage movement.
    else:
        #% Is the Otsu threshold large enough?
        searchDiffs = frameDiffs[prevPeakEndI:];
        otsuThr = graythreshmat(searchDiffs);
        isOtsu = otsuThr > gOtsuThr; #% false if no global Otsu
        
        if not isOtsu:
            #% Does the Otsu threshold separate the 99% of the small frame
            #% differences from the large ones? And, if there is a global small
            #% threshold, is the Otsu threshold larger?
            smallDiffs, smallThr = _get_small_otsu(frameDiffs, gOtsuThr)
            isOtsu = (smallDiffs.size>0) & \
            np.any(~np.isnan(smallDiffs)) & \
            (otsuThr >= smallThr);
            isOtsu = isOtsu & (np.isnan(gSmallThr) | (otsuThr > gSmallThr))
            
            #% Does the global Otsu threshold pull out any peaks?
            if not isOtsu:
                if not np.isnan(gOtsuThr) and (np.sum(searchDiffs > gOtsuThr) > 1):
                    otsuThr = gOtsuThr;
                    isOtsu = True;
        
            
       
        #% Are there any large frame difference past the last stage movement?
        isExtraPeaks = False;
        if not isOtsu:
            peakI = frameDiffs.size;
            peakBackEndI = frameDiffs.size-1
        #% There are too many large frame-difference peaks.
        else:
            _, indices = maxPeaksDistHeight(searchDiffs, maxMoveFrames, otsuThr);
            isExtraPeaks = len(indices)>0;
            
            #% Find the first large peak past the last stage movement.
            i = prevPeakEndI;
            while (i < frameDiffs.size-1) and \
            (np.isnan(frameDiffs[i]) or (frameDiffs[i] < otsuThr)):
                i = i + 1;
            peakI = i;
            
            #% Find a temporary back end for this large peak.
            #% Note: this peak may serve as its own temporary back end.
            startI = max(peakI - maxMoveFrames, prevPeakEndI);
            
            dd = frameDiffs[startI:peakI+1][::-1]
            i = np.nanargmin(dd)
            minDiff = dd[i]
            peakBackEndI = peakI - i + 1; #% we flipped to choose the last min
            
            #% If the temporary back end's frame difference is small, try to
            #% push the back end forwards (closer to the stage movement).
            if minDiff <= prevSmallThr:
                i = peakI - 1;
                while i > startI:
                    if frameDiffs[i] <= prevSmallThr:
                        peakBackEndI = i;
                        break;
                    
                    i = i - 1;
                
            #% If the temporary back end's frame difference is large, try to
            #% push the back end backwards (further from the stage movement).
            elif minDiff >= min(otsuThr, gOtsuThr) or \
            ((minDiff > gSmallThr) and (peakBackEndI > startI) and \
             np.all(np.isnan(frameDiffs[startI:(peakBackEndI - 1)]))):
                peakBackEndI = startI;
        
        #% Compute a threshold for stage movement.
        smallDiffs = frameDiffs[prevPeakEndI:peakBackEndI+1];
        smallThr = np.nanmean(smallDiffs) + 3 * np.nanstd(smallDiffs, ddof=1);
        if np.isnan(smallThr):
            smallThr = prevSmallThr;
        
        
        
        #% Find the front end for the last logged stage movement.
        i = prevPeakI;
        while (i < peakI) and (i < frameDiffs.size-1) and  \
        (np.isnan(frameDiffs[i]) or (frameDiffs[i] > smallThr)) and \
        (np.isnan(frameDiffs[i + 1]) or (frameDiffs[i + 1] > smallThr)):
            i = i + 1;
        
        movesI[-1,1] = i;
        prevPeakEndI = i-1;
        
        
        #% Mark the last logged stage movement.
        if movesI.shape[0] == 1:
            frames[:movesI[-1, 1]] = True
        else:
            frames[movesI[-1,0]:movesI[-1,1]] = True
        
        #% Are there any large frame-difference peaks after the last logged
        #% stage movement?
        if isExtraPeaks:
            pass
            #warning('findStageMovement:TooManyPeaks', ...
            #    ['There are, approximately, ' num2str(length(indices)) ...
            #    ' large frame-difference peaks after the last stage' ...
            #    ' movement ends at ' num2str((movesI(end,2) - 1)/ fps, '%.3f') ...
            #    ' seconds (frame ' num2str(movesI(end,2) - 1) ')']);
        #% Find the back end for logged stage movements.
        i = peakI - 1;
        while (i > prevPeakEndI) and (np.isnan(frameDiffs[i]) or \
                (frameDiffs[i] > smallThr)):
            i = i - 1;
        
        
        
        movesI = np.vstack((movesI, (i+1, frameDiffs.size+1)))
        frames[movesI[-1,0]:] = True;
        
    #% Are any of the stage movements considerably small or large?
    if isExtraPeaks:
        #% Compute the stage movement sizes.
        movesI = movesI[:i, :]
        moveSizes = np.zeros((movesI.shape[0],1));
        for j in range(1, movesI.shape[0]-1):
            moveDiffs = frameDiffs[movesI[j,0]:movesI[j,1]];
            moveSizes[j] = np.nansum(moveDiffs);
        
        #% Compute the statistics for stage movement sizes.
        meanMoveSize = np.nanmean(moveSizes[1:]);
        stdMoveSize = np.nanstd(moveSizes[1:], ddof=1);
        smallMoveThr = meanMoveSize - 2.5 * stdMoveSize;
        largeMoveThr = meanMoveSize + 2.5 * stdMoveSize;
        
        #% Are any of the stage movements considerably small or large?
        for i in range(1, movesI.shape[0]-1):
            #% Is the stage movement small?
            if moveSizes[i] < smallMoveThr:
                before_f = movesI[i,0] - 1
                after_f = movesI[i,1] - 1
                #% Report the warning.
                dd = ['ShortMove',
                     'Stage movement {}'.format(i),
                     'at media time {:.3}'.format(mediaTimes[i]),
                     'seconds (frame {}),'.format(int(round(mediaTimes[i] * fps))),
                     'spanning from {:.3} seconds (frame {})'.format(before_f / fps, before_f),
                     'to {:.3} seconds (frame {}),'.format(after_f/fps, after_f),
                     'is considerably small'
                     ]
                warnings.warn(' '.join(dd))
                
            #% Is the stage movement large?
            elif moveSizes[i] > largeMoveThr:
                before_f = movesI[i,0] - 1
                after_f = movesI[i,1] - 1
                #% Report the warning.
                dd = ['LongMove',
                     'Stage movement {}'.format(i),
                     'at media time {:.3}'.format(mediaTimes[i]),
                     'seconds (frame {}),'.format(int(round(mediaTimes[i] * fps))),
                     'spanning from {:.3} seconds (frame {})'.format(before_f / fps, before_f),
                     'to {:.3} seconds (frame {}),'.format(after_f/fps, after_f),
                     'is considerably large'
                     ]
                warnings.warn(' '.join(dd))
        
        
        
    return frames, movesI, locations

#%%
def shift2video_ref(is_stage_move, movesI, stage_locations, video_timestamp_ind):
    stage_vec = np.full((is_stage_move.size,2), np.nan);
    
    
    if len(movesI) <= 1 and np.all(movesI==0):
        #%there was no movements
        stage_vec[:,0] = stage_locations[:, 0];
        stage_vec[:,1] = stage_locations[:, 1];
        
    else:
        #%convert output into a vector that can be added to the skeletons file to obtain the real worm displacements
        for kk in range(stage_locations.shape[0]):
            bot = max(0, movesI[kk,1]);
            top = min(is_stage_move.size, movesI[kk+1,0]);
            stage_vec[bot:top, 0] = stage_locations[kk,0];
            stage_vec[bot:top, 1] = stage_locations[kk,1];
        
    #%the nan values must match the spected video motions
    #assert(all(isnan(stage_vec(:,1)) == is_stage_move))
    
    # prepare vectors to save into the hdf5 file.
    #%Go back to the original movie indexing. I do not want to include the missing frames at this point.
    is_stage_move_d = is_stage_move[video_timestamp_ind].astype(np.int8);
    stage_vec_d = stage_vec[video_timestamp_ind, :];
    
    return stage_vec_d, is_stage_move_d