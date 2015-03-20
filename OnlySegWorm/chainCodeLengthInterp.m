function [interpData indices] = chainCodeLengthInterp(data, lengths, ...
    chainCodeLengths, varargin)
%INTERPDATA Interpolate data values at the requested chain-code lengths.
%
%   [INTERPDATA INDICES] = CHAINCODELENGTHINTERP(DATA, LENGTHS,
%       CHAINCODELENGTHS)
%
%   [INTERPDATA INDICES] = CHAINCODELENGTHINTERP(DATA, LENGTHS,
%       CHAINCODELENGTHS, INDICES)
%
%   Inputs:
%       data             - the original data values
%       lengths          - the lengths at which to interpolate data values
%       chainCodeLengths - an ascending array of chain code lengths
%                          Note: the chain code lengths must increase at
%                          every successive index
%       indices          - the indices for the elements closest to the
%                          desired lengths; if empty, the indices are
%                          computed using chainCodeLength2Index
%
%   Outputs:
%       interpData       - the interpolated data values
%       indices          - the indices for the elements closest to the
%                          desired lengths
%
% See also COMPUTECHAINCODELENGTHS, CHAINCODELENGTH2INDEX
%
%
% © Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Is the data 2 dimensional?
if ndims(data) > 2 || (size(data, 1) > 2 && size(data, 2) > 2)
    error('chainCodeLengthInterp:dataSize', ...
        'The input data must be, at most, 2 dimensional')
end

% Transpose the data.
isTransposed = false;
if size(data, 2) > 2
    data = data';
    isTransposed = true;
end

% Are there indices for the lengths?
if length(varargin) == 1
    indices = varargin{1};
else
    indices = chainCodeLength2Index(lengths, chainCodeLengths);
end

% Interpolate the in-between data.
sqrt2 = sqrt(2);
interpData = data(indices,:);
for i = 1:numel(lengths)
    
    % Compute the difference between the indexed and requested length.
    if indices(i) == 1 && lengths(i) > chainCodeLengths(end)
        dLength = chainCodeLengths(1) - lengths(i) + ...
            chainCodeLengths(end);
    elseif indices(i) == length(chainCodeLengths) && ...
            lengths(i) < chainCodeLengths(1)
        dLength = -lengths(i);
    else
        dLength = chainCodeLengths(indices(i)) - lengths(i);
    end
    
    % The requested length is between the previous and current index.
    if dLength > 0
        nextI = indices(i) - 1;
        if nextI < 1
            
            % Is the data circularly connected?
            if chainCodeLengths(1) > 0
                nextI = nextI + length(chainCodeLengths);
            else
                dLength = 0;
            end
        end
        
    % The requested length is between the current and next index.
    elseif dLength < 0
        nextI = indices(i) + 1;
        if nextI > length(chainCodeLengths)
        
            % Is the data circularly connected?
            if chainCodeLengths(1) > 0
                nextI = nextI - length(chainCodeLengths);
            else
                dLength = 0;
            end
        end
    end
    
    % Interpolate the data.
    if dLength ~= 0
        
        % The data is 1 dimensional.
        dLength = abs(dLength);
        if size(data, 2) == 1
            
            % Interpolate the data by adding the previous and next data
            % values weighted by their distance from the requested length.
            dNextLength = abs(chainCodeLengths(nextI) - lengths(i));
            interpData(i) = (dNextLength * data(indices(i)) + ...
                dLength * data(nextI)) / (dLength + dNextLength);
            
        % The data is 2 dimensional.
        else
            
            % Interpolate the data by:
            %
            % 1. Compute the line between the previous and next data values.
            % 2. Compute the location of the requested length on this line.
            % 3. The coordinates for the location of the requested length
            %    correspond to the interpolated data values.
            %
            % Note: the data values at the requested length lie on the line
            % separating the closest data values (index = indices(i)) and
            % the second closest data values, at the next or previous index
            % (index = nextI). Now, we need to add the magnitude of the
            % difference between the requested and real distance (dLength)
            % to the closest data values (indices(i)), going in a line
            % towards the second closest data values (nextI). Therefore, we
            % need to solve the difference between the requested data
            % values and the closest ones. Remember the requested data
            % values lie on the slope between the closest (indices(i)) and
            % second closest (nextI) data values. Therefore, the slope can
            % be computed from the difference:
            %
            %   dData = data(nextI,:) - data(indices(i),:)
            %
            % As the slope m:
            %
            %   dData(1) = m * dData(2)
            %
            % We want to solve:
            %
            %   dLength = sqrt(dData(1)^2 + dData(2)^2)
            %
            % Plugging in the slope m and re-arranging the inequalities,
            % we get the data offsets:
            %
            %   offData(1) = dLength / sqrt(1 + (dData(2) / dData(1)) ^ 2)
            %   offData(2) = dLength / sqrt(1 + (dData(1) / dData(2)) ^ 2)
            %
            % Finally, signing the offsets and adding them to the previous
            % data values, we solve the interpolated data values as:
            %
            %   interpData = offData .* sign(dData) + data(prevI,:)
            dData = data(nextI,:) - data(indices(i),:);
            if any(dData == 0)
                interpData(i,:) = dLength .* sign(dData) + ...
                    data(indices(i),:);
            elseif all(abs(dData) == 1)
                interpData(i,:) = (dLength / sqrt2) .* dData + ...
                    data(indices(i),:);
            else
                offData1 = dLength / sqrt(1 + (dData(2) / dData(1)) ^ 2);
                offData2 = dLength / sqrt(1 + (dData(1) / dData(2)) ^ 2);
                interpData(i,:) = [offData1 offData2] .* sign(dData) + ...
                    data(indices(i),:);
            end
        end
    end
end

% Transpose the interpolated data.
if isTransposed
    interpData = interpData';
end
end
