function imOut = unwrap2D(imIn)
M = size(imIn,1);
N = size(imIn,2);

xDelta = mod(circshift(imIn,-1,1) - imIn + pi,2*pi) - pi;
xDelta(end,:) = 0;
yDelta = mod(circshift(imIn,-1,2) - imIn + pi,2*pi) - pi;
yDelta(:,end) = 0;

d = (xDelta - circshift(xDelta,1,1)) + (yDelta - circshift(yDelta,1,2));

dBar = dct2(d);

psiBar = dBar./(2*(cos(pi*repmat(1:M,M,1)./M) + cos(pi*repmat((1:N).',1,N)./N) - 2));

imOut = idct2(psiBar);
end