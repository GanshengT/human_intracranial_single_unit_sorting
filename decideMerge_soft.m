function shouldMerge = decideMerge_soft(pc_scores_cluster1, pc_scores_cluster2, ...
    all_spike_times_cluster1, all_spike_times_cluster2)
%% description - Gansheng Tan
% if acg number is small, use a soft bimodal thres, don't merge, if bimodal but refractory, merge!
% if acg number is high, we merge if passing acg check or pass a hard bimodal thres

    total_bin = 400;
    projection_range = [-2 2];
    trough_finding_range = [175, 225];
    bimod_thres = 0.5;
    bimod_thres_soft = 0.6;
    bin_acg = -0.5:0.001:0.5; %0.5s
    refrctory_window = 5; %5ms
    refrctory_window_indices = find((bin_acg >= (-refrctory_window * 1e-3)) & ...
        bin_acg <= (refrctory_window * 1e-3));
    bimod_minimum_factor = 4;  % at least 4 * pc for regression
    acg_minimum_point = 2000;
    ccg_threshold = 0.25;
    acg_threshold = 0.1;
    highly_corr_thres = 0.75;
    cond_thres = 1e12;

    X = [pc_scores_cluster1; pc_scores_cluster2];
    y = [-ones(size(pc_scores_cluster1, 1), 1); ones(size(pc_scores_cluster2, 1), 1)]; % Labels

    n1 = size(pc_scores_cluster1, 1);
    n2 = size(pc_scores_cluster2, 1);
    [R, P] = corr(X');
    
    w = [n2/(n1 + n2) * ones(n1, 1); n1/(n1 + n2) * ones(n2, 1)];

    % check if the number of samples is enough for bimodal calculation (we
    % dont want overfitting)
    % small sample, we check correlation
    if (n1 < (size(pc_scores_cluster1, 2) * bimod_minimum_factor)) || ...
            (n2 < (size(pc_scores_cluster1, 2) * bimod_minimum_factor))
        % check correlation
        highly_corr = mean(mean(R(1:n1, n1+1:n1+n2)));
        if highly_corr > highly_corr_thres
            shouldMerge = true;
        else
            shouldMerge = false;
        end
        return 
    end
    
    % Weighted linear regression to find regression axis
    % the equation is (X'WX) beta = X'Wy, below is the solution
    
    if cond((X' * diag(w) * X)) > cond_thres
        % if singular, compare correlation
        highly_corr = mean(mean(R(1:n1, n1+1:n1+n2)));
        if highly_corr > highly_corr_thres
            bimod = 0; % because they are highly correlated
        else
            bimod = 1;
        end
    else
        u = (X' * diag(w) * X) \ (X' * diag(w) * y);
        x_proj = X * u;
    
        edges = linspace(projection_range(1), projection_range(2), total_bin);
        hist_counts = histcounts(x_proj, edges, 'Normalization', 'probability');
        smooth_hist = imgaussfilt(hist_counts, 4);
        
        % Find bimodality score = 1-max(xmin/x1, xmin/x2)
        [~, imin] = min(smooth_hist(trough_finding_range(1):trough_finding_range(2)));
        trough = mean(smooth_hist(trough_finding_range(1):trough_finding_range(2)));
        imin = imin + trough_finding_range(1) - 1;
        [peak1, ~] = max(smooth_hist(1:imin));
        [peak2, ~] = max(smooth_hist(imin+1:end));
        bimod = 1 - max(trough/peak1, trough/peak2);
    end

    
    % Decide to merge based on bimodality score threshold
    % higher bimod means bimodal
    % if the spikes are from two neurons, we expect ACG for two has lower
    % occurentce in central bin, we only evaluate ACG
    % if ACG do not have
    % a lower central value, then dont merge.
    % if the number of occurrentce is small, we neglect this non-merging
    % criteria.
    combined_spike_time = [all_spike_times_cluster1(:); all_spike_times_cluster2(:)];
    combined_spike_time = sort(combined_spike_time);
    diffST2 = combined_spike_time - combined_spike_time.';
    diffST2 = diffST2(:);
    ACG = histcounts(diffST2, bin_acg);
    if sum(ACG) < acg_minimum_point
        % (this usually happen at the begining of merging)
        % we will use a hard bimod_thres
        % meaning only merge if they are very similar
        if (bimod < bimod_thres)
            shouldMerge = true;
        else
            shouldMerge = false;
        end
    else
        % the operation will result in 0 diagonol
        center_max_ACG = max(ACG(refrctory_window_indices)) - length(diffST2);
        other_indices = true(size(ACG)); % Initialize all as true
        other_indices(refrctory_window_indices) = false;
        other_max_ACG = max(ACG(other_indices));
        if (center_max_ACG / other_max_ACG) > acg_threshold
            shouldMerge = false;
        else
            if (bimod < bimod_thres_soft)
                shouldMerge = true;
            % we pass acg test, then we add a soft bimod_thres
            else
                shouldMerge = false;
            end
        end
    end

    % if (bimod < bimod_thres)
    %     shouldMerge = true;
    %     diffST2 = all_spike_times_cluster2(:) - all_spike_times_cluster1(:).';
    %     diffST2 = diffST2(:);
    %     CCG = histcounts(diffST2, bin_acg); % Histogram of differences
    %     if sum(CCG) > acg_minimum_point
    %         % if we have acg_minimum_point within -0.5to0.5s, we enable CCG
    %         % criterion
    %         center_max_CCG = max(CCG(refrctory_window_indices));
    %         other_indices = true(size(CCG)); % Initialize all as true
    %         other_indices(refrctory_window_indices) = false;
    %         other_max_CCG = max(CCG(other_indices));
    %         if center_max_CCG / other_max_CCG > ccg_threshold
    %             shouldMerge = false;
    %         end
    %     end
    %     combined_spike_time = [all_spike_times_cluster1(:); all_spike_times_cluster2(:)];
    %     combined_spike_time = sort(combined_spike_time);
    %     diffST2 = combined_spike_time - combined_spike_time.';
    %     diffST2 = diffST2(:);
    %     ACG = histcounts(diffST2, bin_acg);
    %     if sum(ACG) > acg_minimum_point
    %         center_max_ACG = max(ACG(refrctory_window_indices));
    %         other_indices = true(size(ACG)); % Initialize all as true
    %         other_indices(refrctory_window_indices) = false;
    %         other_max_ACG = max(ACG(other_indices));
    %         if center_max_ACG / other_max_ACG > acg_threshold
    %             shouldMerge = false;
    %         end
    %     end
    % else
    %     shouldMerge = false;
    % end
end
