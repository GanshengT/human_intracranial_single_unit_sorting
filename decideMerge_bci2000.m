function shouldMerge = decideMerge_bci2000(pc_scores_cluster1, pc_scores_cluster2, ...
    all_spike_times_cluster1, all_spike_times_cluster2, merge_prctil, retained_coeff, mu)
%% description - Gansheng Tan
% if acg number is small, use a soft bimodal thres, don't merge, if bimodal but refractory, merge!
% if acg number is high, we merge if passing acg check or pass a hard bimodal thres
    rng(0);
    total_bin = 400;
    projection_range = [-2 2];
    % trough_finding_range = [175, 225];
    perm_perctg = 0.2; % at least half the maximum density
    % bimod_thres = 0.6;
    bin_acg = -0.5:0.001:0.5; %0.5s
    refrctory_window = 4; %5ms
    refrctory_window_indices = find((bin_acg >= (-refrctory_window * 1e-3)) & ...
        bin_acg <= (refrctory_window * 1e-3));
    % bimod_minimum_factor = 8;  % at least 20 * pc for regression
    % this parameter should be used together with bimod_thres, smooth
    % factor
    acg_minimum_point = 2000;
    % ccg_threshold = 0.25;
    acg_threshold = 0.25;  % acg means crosscorrelogram
    highly_corr_thres = 0.80;  % used for small clusters
    cond_thres = 1e12;
    n_simulation = 200;
    n_permutation = 1000;

    n1 = size(pc_scores_cluster1, 1);
    n2 = size(pc_scores_cluster2, 1);
    % [R, ~] = corr(X');
    % I had a problem for bimodal projection, as the first clusters is big (messy),
    % the projection is not centered at -1 but centered at 0. Therefore,
    % although the two clusters should not be merged, the bimodal score
    % will be low. I tried the following approach, but it does not solve
    % the problem
    % instead of using weighted least square, we can oversample
    % empirically, we observe
    targetSize = max(n1, n2);
    if n1 < n2
        pc_mean = mean(pc_scores_cluster1);
        pc_cov = cov(pc_scores_cluster1);  
        num_new_samples = targetSize - n1;
        cluster_to_oversample = 1;
    else
        pc_mean = mean(pc_scores_cluster2);
        pc_cov = cov(pc_scores_cluster2); 
        num_new_samples = targetSize - n2;
        cluster_to_oversample = 2;
    end

    observed_bimod = NaN(n_simulation, 1);
    prominence_used = NaN(n_simulation, 1);
    for i_sim = 1:n_simulation
        new_samples = mvnrnd(pc_mean, pc_cov, num_new_samples);
        if cluster_to_oversample == 1
            pc_scores_cluster1_resampled = [pc_scores_cluster1; new_samples];
            X = [pc_scores_cluster1_resampled; pc_scores_cluster2];
            y = [-ones(size(pc_scores_cluster1_resampled, 1), 1); ones(size(pc_scores_cluster2, 1), 1)];
        else
            pc_scores_cluster2_resampled = [pc_scores_cluster2; new_samples];
            X = [pc_scores_cluster1; pc_scores_cluster2_resampled];
            y = [-ones(size(pc_scores_cluster1, 1), 1); ones(size(pc_scores_cluster2_resampled, 1), 1)];
        end
    
        % simulation
        % n_pc = size(pc_scores_cluster1, 2);
        % specificity_scores = zeros(n_simulation, 1);
        % for i_sim = 1:n_simulation
        %     pc_scores_cluster1_sim = zeros(n1, n_pc);
        %     pc_scores_cluster2_sim = zeros(n2, n_pc);
        %     std_dev1 = std(pc_scores_cluster1);
        %     std_dev2 = std(pc_scores_cluster2);
        %     for col = 1:n_pc
        %         pc_scores_cluster1_sim(:, col) = normrnd(0, std_dev1(col), n1, 1);
        %         pc_scores_cluster2_sim(:, col) = normrnd(0, std_dev2(col), n2, 1);
        %     end
        %     X = [pc_scores_cluster1_sim; pc_scores_cluster2_sim];
        %     y = [ones(size(pc_scores_cluster1, 1), 1); 2*ones(size(pc_scores_cluster2, 1), 1)]; % Labels
        %     [idx, ~] = kmeans(X, 2);
        %     cm1 = confusionmat(y, idx); 
        %     cm2 = confusionmat(y, 3 - idx); 
        %     specificity_score1 = (cm1(1, 1) / sum(cm1(:, 1)) + cm1(2, 2) / sum(cm1(:, 2)))/ 2;
        %     specificity_score2 = (cm2(1, 1) / sum(cm2(:, 1)) + cm2(2, 2) / sum(cm2(:, 2)))/ 2;
        %     specificity_scores(end + 1) = max(specificity_score1, specificity_score2);
        % end
    
        % bimodal implementation
        % w = [n2/(n1 + n2) * ones(n1, 1); n1/(n1 + n2) * ones(n2, 1)];
    
        % check if the number of samples is enough for bimodal calculation (we
        % dont want overfitting)
        % small sample, we check correlation
        % if (n1 + n2) < (size(pc_scores_cluster1, 2) * bimod_minimum_factor)
        %     % check correlation
        %     highly_corr = mean(mean(R(1:n1, n1+1:n1+n2)));
        %     if highly_corr > highly_corr_thres
        %         shouldMerge = true;
        %     else
        %         shouldMerge = false;
        %     end
        %     return 
        % end
    
        % Weighted linear regression to find regression axis
        % the equation is (X'WX) beta = X'Wy, below is the solution
        
        % if cond((X' * diag(w) * X)) > cond_thres
        if cond((X' * X)) > cond_thres
            % if singular, compare correlation
            [R, ~] = corr(X');
            highly_corr = mean(mean(R(1:n1, n1+1:n1+n2)));
            if highly_corr > highly_corr_thres
                bimod = 0; % because they are highly correlated
            else
                bimod = 1;
            end
        else
            % u = (X' * diag(w) * X) \ (X' * diag(w) * y);
            u = (X' * X) \ (X' * y);
            x_proj = X * u;
    
            edges = linspace(projection_range(1), projection_range(2), total_bin);
            hist_counts = histcounts(x_proj, edges, 'Normalization', 'probability');
            smooth_hist = imgaussfilt(hist_counts, 4);
    
            % Find bimodality score = 1-max(xmin/x1, xmin/x2)
            % [~, imin] = min(smooth_hist(trough_finding_range(1):trough_finding_range(2)));
            % trough = min(smooth_hist(trough_finding_range(1):trough_finding_range(2)));
            % imin = imin + trough_finding_range(1) - 1;
            % [peak1, ~] = max(smooth_hist(1:imin));
            % [peak2, ~] = max(smooth_hist(imin+1:end));
            % bimod = 1 - max(trough/peak1, trough/peak2);
    
            % refine bimodality score
            % first we calculate the largest prominence
            peakCounts = NaN(n_simulation,1);
            peakDistances = NaN(n_simulation,1);
            troughMean = NaN(n_simulation,1);
            max_prominence = max(smooth_hist);
            min_prominence = 1 / 2/ total_bin;
            currentProminence = max_prominence;
            decrement = (max_prominence - min_prominence) / n_simulation;
            i_count = 1;
            find_peaks_success = false;
            while currentProminence > min_prominence
                [peaks,locs,~,~] = findpeaks(smooth_hist, 'MinPeakProminence', currentProminence);
                numPeaks = numel(peaks);
                peakCounts(i_count) = numPeaks;
                if numPeaks >= 2
                    if numPeaks > 2
                        % can not find two peaksz
                        bimod = 0;
                        break
                    end
                    peakDist = abs(diff(locs));
                    troughLoc = [round(mean(locs) - peakDist / 4) round(mean(locs) + peakDist / 4)];
                    peakDistances(i_count) = peakDist;
                    troughMean(i_count) = mean(smooth_hist(troughLoc));
                    % (total_bin / 2) is theorectical peak distance
                    % if peakDist > (total_bin / 2), we need to enhance the
                    % bimod score, max ensure that it is > 0
                    bimod = max(0, (1 - mean(troughMean(i_count) ./ peaks))) * ...
                        min(1, (peakDist / (total_bin / 2)));
                    find_peaks_success = true;
                    prominence_used(i_sim) = currentProminence;
                    break
                else
                    peakDistances(i_count) = NaN;
                    troughMean(i_count) = NaN;
                end
                currentProminence = currentProminence - decrement;
                i_count = i_count + 1;
            end
            if ~find_peaks_success
                % only one peak
                bimod = 0;
            end
        end
        observed_bimod(i_sim) = bimod;
    end

    % % permutation to get threshold - might fail because if they are from the
    % same neuron, observed bimod score should be within the bimod score distribution from 
    % permutation 
    bimodalScores = zeros(n_permutation, 1);
    num_to_perm = round(perm_perctg * numel(y));

    max_prominence = max(prominence_used, [], 'omitnan');
    min_prominence = min(prominence_used, [], 'omitnan');
    decrement = (max_prominence - min_prominence) / (n_simulation / 10);

    for i_sim = 1:n_permutation
        new_samples = mvnrnd(pc_mean, pc_cov, num_new_samples);
        if cluster_to_oversample == 1
            pc_scores_cluster1_resampled = [pc_scores_cluster1; new_samples];
            X = [pc_scores_cluster1_resampled; pc_scores_cluster2];
            y = [-ones(size(pc_scores_cluster1_resampled, 1), 1); ones(size(pc_scores_cluster2, 1), 1)];
        else
            pc_scores_cluster2_resampled = [pc_scores_cluster2; new_samples];
            X = [pc_scores_cluster1; pc_scores_cluster2_resampled];
            y = [-ones(size(pc_scores_cluster1, 1), 1); ones(size(pc_scores_cluster2_resampled, 1), 1)];
        end
        perm_indices = randperm(length(y), num_to_perm);
        y_perm = y;
        temp = y_perm(perm_indices);
        y_perm(perm_indices) = temp(randperm(length(temp))); 
        u_perm = (X' * X) \ (X' * y_perm);
        x_proj_perm = X * u_perm;

        hist_counts_perm = histcounts(x_proj_perm, edges, 'Normalization', 'probability');
        smooth_hist_perm = imgaussfilt(hist_counts_perm, 4);

        % find peak with various prominence
        i_count = 1;
        find_peaks_success = false;
        currentProminence = max_prominence;
        % because we do not start from the peak, it is possible that we
        % find more than 2 peaks with max_prominence
        [peaks,locs,~,~] = findpeaks(smooth_hist_perm, 'MinPeakProminence', currentProminence);
        numPeaks = numel(peaks);
        if numPeaks > 2
            % do not consider this outlier permutation (very unlikely)
            bimodalScores(i_sim) = NaN;
            continue
        end
        while currentProminence > min_prominence
            [peaks,locs,~,~] = findpeaks(smooth_hist_perm, 'MinPeakProminence', currentProminence);
            numPeaks = numel(peaks);
            if numPeaks >= 2
                if numPeaks > 2
                    % can not find two peaksz
                    bimod = 0;
                    break
                end
                peakDist = abs(diff(locs));
                troughLoc = [round(mean(locs) - peakDist / 4) round(mean(locs) + peakDist / 4)];
                troughMean_perm = mean(smooth_hist(troughLoc));
                % (total_bin / 2) is theorectical peak distance
                % if peakDist > (total_bin / 2), we need to enhance the
                % bimod score, max ensure that it is > 0
                bimod = max(0, (1 - mean(troughMean_perm ./ peaks))) * ...
                    min(1, (peakDist / (total_bin / 2)));
                find_peaks_success = true;
                break
            end
            currentProminence = currentProminence - decrement;
            i_count = i_count + 1;
        end
        if ~find_peaks_success
            % only one peak
            bimod = 0;
        end
        bimodalScores(i_sim) = bimod;
    end
    bimodalScores = sort(bimodalScores, 'ascend');
    bimod_thres = prctile(bimodalScores, merge_prctil);
        

    %     [peaks_perm, locs_perm, ~, ~] = findpeaks(smooth_hist_perm, ...
    %         'MinPeakProminence', currentProminence);
    %     numPeaks_perm = numel(peaks_perm);
    %     if numPeaks_perm >= 2
    %         [sortedPeaks, sortIndex] = sort(peaks_perm, 'descend');
    %         sortedLocs = locs_perm(sortIndex);
    % 
    %         if numel(sortedPeaks) > 2
    %             % Only consider the two most prominent peaks
    %             sortedPeaks = sortedPeaks(1:2);
    %             sortedLocs = sortedLocs(1:2);
    %         end
    %         peakDist_perm = abs(diff(sortedLocs));
    %         troughLoc_perm = [round(mean(locs) - peakDist_perm / 4) round(mean(locs) + peakDist_perm / 4)];
    %         troughMean_perm= mean(smooth_hist_perm(troughLoc_perm));
    %         bimodalScores(i_sim) = max(0, ...
    %             (1 - mean(troughMean_perm ./ sortedPeaks))) * (peakDist_perm / (total_bin / 2));
    %     end
    % end
    


    % % use simulation to determine bimodal threshold
    % n_pc = size(pc_scores_cluster1, 2);
    % 
    % bimodalScores = zeros(n_simulation, 1);
    % if cluster_to_oversample == 1
    %     std_dev1 = std(pc_scores_cluster1_resampled);
    %     std_dev2 = std(pc_scores_cluster2);
    %     for i_sim = 1:n_simulation
    %         pc_scores_cluster1_sim = zeros(size(pc_scores_cluster1_resampled, 1), n_pc);
    %         pc_scores_cluster2_sim = zeros(n2, n_pc);
    %         for col = 1:n_pc
    %             pc_scores_cluster1_sim(:, col) = normrnd(0, std_dev1(col), ...
    %                 size(pc_scores_cluster1_resampled, 1), 1);
    %             pc_scores_cluster2_sim(:, col) = normrnd(0, std_dev2(col), n2, 1);
    %         end
    %         X = [pc_scores_cluster1_sim; pc_scores_cluster2_sim];
    %         y = [-ones(size(pc_scores_cluster1_sim, 1), 1); ...
    %             ones(size(pc_scores_cluster2_sim, 1), 1)];
    %         % w = [n2/(n1 + n2) * ones(n1, 1); n1/(n1 + n2) * ones(n2, 1)];
    %         u = (X' * X) \ (X' * y);
    %         % u = (X' * diag(w) * X) \ (X' * diag(w) * y);
    %         x_proj = X * u;
    %         hist_counts = histcounts(x_proj, edges, 'Normalization', 'probability');
    %         smooth_hist = imgaussfilt(hist_counts, 4);
    % 
    %         % Find bimodality score = 1-max(xmin/x1, xmin/x2)
    %         [~, imin] = min(smooth_hist(trough_finding_range(1):trough_finding_range(2)));
    %         trough = min(smooth_hist(trough_finding_range(1):trough_finding_range(2)));
    %         imin = imin + trough_finding_range(1) - 1;
    %         [peak1, ~] = max(smooth_hist(1:imin));
    %         [peak2, ~] = max(smooth_hist(imin+1:end));
    %         bimodalScores(i_sim) = 1 - max(trough/peak1, trough/peak2);
    %     end
    % 
    % else
    %     std_dev1 = std(pc_scores_cluster1);
    %     std_dev2 = std(pc_scores_cluster2_resampled);
    %     for i_sim = 1:n_simulation
    %         pc_scores_cluster1_sim = zeros(n1, n_pc);
    %         pc_scores_cluster2_sim = zeros(size(pc_scores_cluster2_resampled, 1), n_pc);
    %         for col = 1:n_pc
    %             pc_scores_cluster1_sim(:, col) = normrnd(0, std_dev1(col), n1, 1);
    %             pc_scores_cluster2_sim(:, col) = normrnd(0, std_dev2(col), ...
    %                 size(pc_scores_cluster2_resampled, 1), 1);
    %         end
    %         X = [pc_scores_cluster1_sim; pc_scores_cluster2_sim];
    %         y = [-ones(size(pc_scores_cluster1_sim, 1), 1); ...
    %             ones(size(pc_scores_cluster2_sim, 1), 1)];
    %         % w = [n2/(n1 + n2) * ones(n1, 1); n1/(n1 + n2) * ones(n2, 1)];
    %         u = (X' * X) \ (X' * y);
    %         % u = (X' * diag(w) * X) \ (X' * diag(w) * y);
    %         x_proj = X * u;
    %         hist_counts = histcounts(x_proj, edges, 'Normalization', 'probability');
    %         smooth_hist = imgaussfilt(hist_counts, 4);
    % 
    %         % Find bimodality score = 1-max(xmin/x1, xmin/x2)
    %         [~, imin] = min(smooth_hist(trough_finding_range(1):trough_finding_range(2)));
    %         trough = min(smooth_hist(trough_finding_range(1):trough_finding_range(2)));
    %         imin = imin + trough_finding_range(1) - 1;
    %         [peak1, ~] = max(smooth_hist(1:imin));
    %         [peak2, ~] = max(smooth_hist(imin+1:end));
    %         bimodalScores(i_sim) = 1 - max(trough/peak1, trough/peak2);
    %     end
    % end
    % 
    % bimod_thres = prctile(bimodalScores, merge_prctil);

    % Decide to merge based on bimodality score threshold
    % higher bimod means bimodal
    % if the spikes are from two neurons, we expect ACG for two has lower
    % occurentce in central bin, we only evaluate ACG
    % if ACG do not have
    % a lower central value, then dont merge.
    % if the number of occurrentce is small, we neglect this non-merging
    % criteria.

    diffST2 = all_spike_times_cluster1(:) - all_spike_times_cluster2(:).';
    upperTriIndices = triu(true(size(diffST2)), 1); 
    diffST2Upper = diffST2(upperTriIndices); 
    ACG = histcounts(diffST2Upper, bin_acg);
    if (sum(ACG) < acg_minimum_point) || isempty(ACG(refrctory_window_indices))
        % (this usually happen at the begining of merging)
        % we will use a hard bimod_thres
        % meaning only merge if they are very similar
        if (median(observed_bimod) <= bimod_thres)
            % use = for cases where observed_bimod = 0;
            shouldMerge = true;
        else
            shouldMerge = false;
        end
    else
        center_ACG = mean(ACG(refrctory_window_indices));
        other_indices = true(size(ACG)); % Initialize all as true
        other_indices(refrctory_window_indices) = false;
        other_max_ACG = max(ACG(other_indices));
        if (center_ACG / other_max_ACG) > acg_threshold
            shouldMerge = false;
        else
            if (median(observed_bimod) <= bimod_thres)
                shouldMerge = true;
            % we pass acg test, then we add a soft bimod_thres
            else
                shouldMerge = false;
            end
        end
    end

end
