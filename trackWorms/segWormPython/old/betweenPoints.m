% Find pointsI between startI and endI, inclusive.
function points = betweenPoints(pointsI, startI, endI)
if startI < endI
    points = pointsI >= startI & pointsI <= endI;
else
    points = pointsI >= startI | pointsI <= endI;
end
end

