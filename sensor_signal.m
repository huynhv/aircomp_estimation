function s = sensor_signal(t)
    T_s = 1;
    s = sin((pi/T_s)*t);
    % s = ones(size(t));
    s(or(t<0,t>T_s)) = 0;
    % s(t==0) = s(t==0)*0.5;
    % s(t == T_s) = s(t == T_s)*0.5;
end
