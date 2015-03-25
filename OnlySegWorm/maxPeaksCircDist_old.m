function [peaks, indices] = maxPeaksCircDist(x, dist, varargin)
%MAXPEAKSCIRCDIST Find the maximum peaks in a circular vector. The peaks
%are separated by, at least, the given distance.
%
%   [PEAKS INDICES] = MAXPEAKSCIRCDIST(X, DIST)
%
%   [PEAKS INDICES] = MAXPEAKSCIRCDIST(X, DIST, CHAINCODELENGTHS)
%
%   Inputs:
%       x                - the vector of values
%       dist             - the minimum distance between peaks
%       chainCodeLengths - the chain-code length at each index;
%                          if empty, the array indices are used instead
%
%   Outputs:
%       peaks   - the maximum peaks
%       indices - the indices for the peaks
%
%   See also MINPEAKSCIRCDIST, CIRCCOMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are there chain-code lengths?
if length(varargin) == 1
    chainCodeLengths = varargin{1};
else
    chainCodeLengths = [];
end

% Use the array indices for length.
if isempty(chainCodeLengths)
    chainCodeLengths = 1:length(x);
end

% Is the vector larger than the search window?
winSize = 2 * dist + 1;
if chainCodeLengths(end) < winSize
    [peaks, indices] = max(x);
    return;
end

% Initialize the peaks and indices.
wins = ceil(length(x) / winSize);
peaks = zeros(wins, 1); % pre-allocate memory
indices = zeros(wins, 1); % pre-allocate memory

% Search for peaks.
im = 0; % the last maxima index
ie = 0; % the end index for the last maxima's search window
ip = 1; % the current, potential, max peak index
p = x(ip); % the current, potential, max peak value
i = 2; % the vector index
j = 1; % the recorded, maximal peaks index
while i <= length(x)
    
    % Found a potential peak.
    if x(i) > p
        ip = i;
        p = x(i);
    end
    
    % Test the potential peak.
    if chainCodeLengths(i) - chainCodeLengths(ip) >= dist || i == length(x)
        
        % Check the untested values next to the previous maxima.
        if im > 0 && chainCodeLengths(ip) - chainCodeLengths(im) <= 2 * dist
            
            % Check the untested values next to the previous maxima. 
            isMax = true;
            k = ie;
            while isMax && k > 0 && ...
                    chainCodeLengths(ip) - chainCodeLengths(k) < dist
                
                % Is the previous peak larger?
                if x(ip) <= x(k)
                    isMax = false;
                end
                
                % Advance.
                k = k - 1;
            end
            
            % Record the peak.
            if isMax
                indices(j) = ip;
                peaks(j) = p;
                j = j + 1;
            end
            
            % Record the maxima.
            im = ip;
            ie = i;
            ip = i;
            p = x(ip);
            
        % Record the peak.
        else
            indices(j) = ip;
            peaks(j) = p;
            j = j + 1;
            im = ip;
            ie = i;
            ip = i;
            p = x(ip);
        end
    end
        
    % Advance.
    i = i + 1;
end

% Collapse any extra memory.
indices(j:end) = [];
peaks(j:end) = [];

% If we have two or more peaks, we have to check the start and end for mistakes.
if j > 2
    
    % If the peaks at the start and end are too close, keep the largest or
    % the earliest one.
    if (chainCodeLengths(indices(1)) + chainCodeLengths(end) - ...
            chainCodeLengths(indices(end))) < dist
        if peaks(1) <= peaks(end)
            indices(1) = [];
            peaks(1) = [];
        else
            indices(end) = [];
            peaks(end) = [];
        end
        
    % Otherwise, check any peaks that are too close to the start and end.
    else
    
        % If we have a peak at the start, check the wrapping portion just
        % before the end.
        k = length(x);
        while chainCodeLengths(indices(1)) + ...
                chainCodeLengths(length(x)) - chainCodeLengths(k) < dist
            
            % Remove the peak.
            if peaks(1) <= x(k)
                indices(1) = [];
                peaks(1) = [];
                break;
            end
            
            % Advance.
            k = k - 1;
        end
            
        % If we have a peak at the end, check the wrapping portion just
        % before the start.
        k = 1;
        while chainCodeLengths(length(x)) - ...
                chainCodeLengths(indices(end)) + chainCodeLengths(k) < dist
            
            % Remove the peak.
            if peaks(end) < x(k)
                indices(end) = [];
                peaks(end) = [];
                break;
            end
            
            % Advance.
            k = k + 1;
        end
    end
end
end