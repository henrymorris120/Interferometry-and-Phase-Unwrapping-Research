
%inputs: lambda is the wavelength, phi and eta are aperture position angles,
% delta is the phi_f-phi_g, interfergram is matrix of the heights we get

function [x_coordinates,y_coordinates,heights] = calculateTerrainHeight(lambda,phi,delta,eta, interferogram, x1,y1); 
x_coordinates=zeros(size(interferogram));
y_coordinates=zeros(size(interferogram));
heights=zeros(size(interferogram));
A=[1,0,-tand(eta);0,1,-tand(phi);0,0,1];

for i= 1:length(x1)
    for j= 1:length(y1)
        B=[x1(i);y1(j);(lambda/(4*pi))*(cosd(phi)/delta)*interferogram(i,j)];
        out=A\B;
        x_coordinates(i,j)=out(1);
        y_coordinates(i,j)=out(2);
        heights(i,j)=out(3);
    end
end


