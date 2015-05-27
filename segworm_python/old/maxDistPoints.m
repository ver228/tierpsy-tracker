% Find pointsI whose index distance from oppPointsI exceeds maxDistI.
function points = maxDistPoints(pointsI, oppPointsI, maxDistI, contour)

% How close are the points?
points = false(length(pointsI),1);
for i = 1:length(pointsI)
    if pointsI(i) > oppPointsI(i)
        
        % The points exceed the threshold.
        if maxDistI <= min(pointsI(i) - oppPointsI(i), ...
                oppPointsI(i) + size(contour, 1) - pointsI(i))
            points(i) = true;
        end
        
        % The points exceed the threshold.
    elseif maxDistI <= min(oppPointsI(i) - pointsI(i), ...
            pointsI(i) + size(contour, 1) - oppPointsI(i))
        points(i) = true;
    end
end
end
