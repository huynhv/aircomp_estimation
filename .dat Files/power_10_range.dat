% POWER_10_RANGE Returns the powers of 10 that are between the specified range.
function [out] = power_10_range(min_val, max_val)
    min_exp = ceil(log10(min_val));
    max_exp = floor(log10(max_val));
    out = 10.^(min_exp:max_exp);
end