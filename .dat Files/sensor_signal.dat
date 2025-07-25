% SENSOR_SIGNAL computes the sensor signal defined in the paper.
function s = sensor_signal(t)
    T_s = 1;
    s = sin(pi*t);
    % s = ones(size(t));
    % s(or(t<0,t>T_s)) = 0;
    s(t<0 | t>T_s) = 0;
    % s(t==0) = s(t==0)*0.5;
    % s(t == T_s) = s(t == T_s)*0.5;
end
