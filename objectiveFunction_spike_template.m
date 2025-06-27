function cost = objectiveFunction_spike_template(A, b, x, lambda)
    % Least squares term
    leastSquares = 0.5 * norm(A*x' - b)^2;
    lassoPenalty = lambda * norm(x', 1);
    % diffs = diff(x');
    % fusedLassoPenalty = lambda * norm(diffs, 1);
      cost = leastSquares + lassoPenalty;
end