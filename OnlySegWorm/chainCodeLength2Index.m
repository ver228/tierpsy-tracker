function indices = chainCodeLength2Index(lengths, chainCodeLengths)
%CHAINCODELENGTH2INDEX Translate a length into an index. The index
%   represents the numerically-closest element to the desired length in
%   an ascending array of chain code lengths.
%
%   INDICES = CHAINCODELENGTH2INDEX(LENGTHS, CHAINCODELENGTHS)
%
%   Inputs:
%       lengths          - the lengths to translate into indices
%       chainCodeLengths - an ascending array of chain code lengths
%                          Note: the chain code lengths must increase at
%                          every successive index
%
%   Output:
%       indices - the indices for the elements closest to the desired
%                 lengths
%
% See also COMPUTECHAINCODELENGTHS, CIRCCOMPUTECHAINCODELENGTHS
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

indices = nan(size(lengths));
% Check the lengths.
% Note: circular chain-code lengths are minimally bounded at 0 and
% maximally bounded at the first + last lengths.
if any(lengths < 0)
    error('chainCodeLength2Index:TooShort', ...
        'The lengths cannot be negative');
end

if any(lengths > chainCodeLengths(1) + chainCodeLengths(end))
    
    error('chainCodeLength2Index:TooLong', ...
        ['The lengths cannot be greater than ', ...
        num2str(chainCodeLengths(end))]);
end

if any(isnan(chainCodeLengths)) || any(isnan(lengths))
     error('chainCodeLength2Index:NaN', ...
         'chainCodeLengths or lenghts contain NaN elements');
end

% Go through the lengths.
%indices = zeros(size(lengths));
for i = 1:numel(lengths)

    % Is the length too small?
    if lengths(i) < chainCodeLengths(1)
        
        % Find the closest index.
        if lengths(i) / chainCodeLengths(1) < .5
            indices(i) = length(chainCodeLengths);
        else
            indices(i) = 1;
        end
        
    % Is the length too big?
    elseif lengths(i) > chainCodeLengths(end)
        
        % Find the closest index.
        if (lengths(i) - chainCodeLengths(end)) / chainCodeLengths(1) < .5
            indices(i) = length(chainCodeLengths);
        else
            indices(i) = 1;
        end

    % Find the closest index.
    else
        
        % Try jumping to just before the requested length.
        % Note: most chain-code lengths advance by at most sqrt(2) at each
        % index. But I don't trust IEEE division so I use 1.5 instead.
        j = round(lengths(i) / 1.5) + 1;
        
        % Did we jump past the requested length?
        if j > length(chainCodeLengths) || lengths(i) < chainCodeLengths(j)
            j = 1;
        end
        
        % find the closest index.
        distJ = abs(lengths(i) - chainCodeLengths(j));
        while j < length(chainCodeLengths)
            
            % Is this index closer than the next one?
            % Note: overlapping points have equal distances. Therefore, if
            % the distances are equal, we advance.
            distNextJ = abs(lengths(i) - chainCodeLengths(j + 1));
            if distJ < distNextJ
                break;
            end
            
            % Advance.
            distJ = distNextJ;
            j = j + 1;
        end
        
        % Record the closest index.
        indices(i) = j;
    end
end
end
