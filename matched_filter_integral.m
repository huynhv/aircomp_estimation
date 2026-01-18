% MATCHED_FILTER_INTEGRAL Returns the matched filter output of tempA and tempB,
% which is computed in a vectorized manner.
function [out] = matched_filter_integral(tempA, tempB, channel_gains, K, S, nTrials, nDeployments, dt)
    tempC = pagemtimes(pagetranspose(tempA),tempB);

    diag_idx = sub2ind(size(tempC),1:S,1:S);
    KTD = K*nTrials*nDeployments;
    all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,K,nTrials,nDeployments);

    tempD = tempC(all_diag_idx)*dt;
    out = pagemtimes(tempD,reshape(channel_gains,S,1,1,1,nDeployments));
end
