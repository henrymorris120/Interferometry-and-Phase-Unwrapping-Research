function imOut = downsample(imIn,r)
[N1,N2] = size(imIn);

sample_x = r+1:2*r+1:N1;
sample_y = r+1:2*r+1:N2;

M1 = length(sample_x);
M2 = length(sample_y);

imOut = zeros(M1,M2);

for ii = 1:M1
    for jj = 1:M2
        if and(ii<M1,jj<M2)
            imOut(ii,jj) = sum(imIn(sample_x(ii)-r:sample_x(ii)+r,sample_y(jj)-r:sample_y(jj)+r),'all');
        elseif and(ii>=M1,jj<M2)
            imOut(ii,jj) = sum(imIn(sample_x(ii)-r:end,sample_y(jj)-r:sample_y(jj)+r),'all');
        elseif and(ii<M1,jj>=M2)
            imOut(ii,jj) = sum(imIn(sample_x(ii)-r:sample_x(ii)+r,sample_y(jj)-r:end),'all');
        else
            imOut(ii,jj) = sum(imIn(sample_x(ii)-r:end,sample_y(jj)-r:end),'all');
        end
    end
end


end