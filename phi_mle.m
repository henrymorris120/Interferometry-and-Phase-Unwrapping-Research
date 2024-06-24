function interferogram_MLE = phi_mle(pass_a, pass_b, neighborhood_size)
% Inputs: pass_a - Complex data for image with M x N pixels at pass a
%         pass_b - Complex data for image with M x N pixels at pass b
%                  NOTE: for now, need to be M x N, same size as pass_a
%         neighborhood_size - size of neighborhood n around each pixel to
%                             take maximum likelihood estimate
% Outputs: interferogram - interferogram between pass a and pass b using
%                          maximum likelihood estimator (MLE) over
%                          neighborhood size

% Determine size of pass_a
M = size(pass_a,1);
N = size(pass_a,2);

% Create interferogram matrix
interferogram_MLE = zeros(M, N);

% For each pixel, calculate the MLE over neighborhood size
for ii = 1:M            % cycle through rows
    for jj = 1:N        % cycle through columns
        % Create neighborhoods for each pixel
        neighborhood_a = pass_a(max(ii - neighborhood_size, 1):min(ii + neighborhood_size, M), ...
            max(jj - neighborhood_size, 1):min(jj + neighborhood_size, N));
        neighborhood_b = pass_b(max(ii - neighborhood_size, 1):min(ii + neighborhood_size, M), ...
            max(jj - neighborhood_size, 1):min(jj + neighborhood_size, N));

        % Resize neighborhoods to be vectors
        neighborhood_a_reshape = reshape(neighborhood_a, [], 1);
        neighborhood_b_reshape = reshape(neighborhood_b, [], 1);

        % a
        interferogram_MLE(ii,jj) = angle(conj(neighborhood_a_reshape).' * neighborhood_b_reshape);
    end
end

end