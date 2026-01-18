% GET_TIME_DOMAIN Returns the time-domain matrix of a given frequency domain
% vector.
function [Qn,Qn_matrix] = get_time_domain(mag_sqr_H, dt, nDeployments)
    mag_sqr_H_2 = [mag_sqr_H(1,1,:) mag_sqr_H(1,2:end,:) fliplr(conj(mag_sqr_H(1,2:end,:)))];
    Qn = ifft(mag_sqr_H_2/dt,[],2);
    L = floor(size(Qn,2)/2);
    mirrored_Qn = [Qn(1,L+2:end,:), Qn(1,1:L+1,:)];
    Qn_matrix = zeros(L+1,L+1,nDeployments);
    for i = 1:(L+1)
        Qn_matrix(i,:,:) = mirrored_Qn(1,L+1-(i-1):end-(i-1),:);
    end
end