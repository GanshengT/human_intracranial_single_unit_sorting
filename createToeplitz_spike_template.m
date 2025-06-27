function A = createToeplitz_spike_template(template, m)
    paddedTemplate = [template; zeros(m - length(template), 1)];
    c = paddedTemplate; % first column
    r = [c(1), zeros(1, m-1)];
    A = toeplitz(c, r);
end

% % to verify
% A_combined = [];
% template = [0 1 0]'; template1 = [1 2 0]'; A = createToeplitz_spike_template(template, 10);A_combined = [A_combined, A]; A = createToeplitz_spike_template(template1, 10);
% A_combined = [A_combined, A];
% x =zeros(20, 1); x(1)=1;         result=A_combined*x; figure; plot(result)
% x =zeros(20, 1); x(11)=1;         result=A_combined*x; figure; plot(result)
% x =zeros(20, 1); x(11)=1;   x(1)=1;      result=A_combined*x; figure; plot(result)