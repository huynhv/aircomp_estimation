% QUANTIZE_COMP Returns the quantized sensor chanel phase.
function [qg,cq] = quantize_comp(g,M)
    phi = angle(g);
    phi = phi + (2*pi)*(phi<0);
    
    all_cq = 0;

    for l = 1:M
        if l == 1
            cq_temp = exp(-(l-1)*1i*2*pi/M) * ((phi > (2*M-1)*pi/M) | (phi <= pi/M));
        else
            cq_temp = exp(-(l-1)*1i*2*pi/M) * ((phi > (2*(l-1)-1)*pi/M) & (phi <= (2*l-1)*pi/M));
        end
        all_cq = all_cq + cq_temp;
    end
    qg = g .* all_cq;
    cq = all_cq;
end
    