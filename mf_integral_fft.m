% mf_integral_fft Returns the matched filter output of tempA and tempB using the FFT.
function [out_tensor, ui_tensor] = mf_integral_fft(A_tensor, B_tensor, ci, ai, K, dt, Tp)
    % tempC = pagemtimes(pagetranspose(tempA),tempB);
    % 
    % diag_idx = sub2ind(size(tempC),1:S,1:S);
    % KTD = K*nTrials*nDeployments;
    % all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,K,nTrials,nDeployments);
    % 
    % ui = tempC(all_diag_idx)*dt;
    % out = pagemtimes(ui,reshape(channel_gains,S,1,1,1,nDeployments));

    N = size(A_tensor,1) + size(B_tensor,1) - 1;
    Af = fft(A_tensor, N, 1);
    Bf = fft(B_tensor, N, 1);
    Y_tensor =  dt*ifft(Af .* Bf, [], 1);
    ui_tensor = Y_tensor(Tp/dt + 1:(Tp/dt + K),:,:,:); % need to make sure index is integer multiple
    out_tensor = sum(ui_tensor .* ci .* ai, 2);
    disp('')
end