% ZERIFNAN Returns 0 if x is NaN
function y = zeroIfnan(x)
   y = x;
   y(isnan(x)) = 0;
end