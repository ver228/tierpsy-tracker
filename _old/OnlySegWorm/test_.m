[skeleton, cWidths] = linearSkeleton(headI, tailI, lfCMinP, lfCMinI, ...
        lfCMaxP, lfCMaxI, contour, wormSegLength, cCCLengths);
%{    
figure 
hold on
plot(skeleton(:,1), skeleton(:,2),'.-g')
plot(skeletonMex(:,1), skeletonMex(:,2), 'o-r')
%plot(oSkeleton(:,1), oSkeleton(:,2), 'x-b')
%}
%figure 
%hold on
%plot(cWidths,'.-g')
%plot(cWidthsMex, 'o-r')
%plot(ocWidths, 'x-b')