% MAGTEN2DB converts a quantity from decimal to dB.
function out = magTen2db(mag_val)
    out = mag2db(sqrt(mag_val));
end