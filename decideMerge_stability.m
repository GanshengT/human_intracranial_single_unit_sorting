function shouldMerge = decideMerge_stability(pc_scores_cluster1, pc_scores_cluster2, ...
    all_spike_times_cluster1, all_spike_times_cluster2, merge_prctil, retained_ceoff, mu)
%% description - Gansheng Tan
% if acg number is small, use a soft bimodal thres, don't merge, if bimodal but refractory, merge!
% if acg number is high, we merge if passing acg check or pass a hard bimodal thres

    num_permutations = 300;
    permutation_proportion = 0.05; % 5%
    % bimod_thres = 0.6;
    bin_acg = -0.5:0.001:0.5; %0.5s
    refrctory_window = 4; %5ms
    refrctory_window_indices = find((bin_acg >= (-refrctory_window * 1e-3)) & ...
        bin_acg <= (refrctory_window * 1e-3));
    % this parameter should be used together with bimod_thres, smooth
    % factor
    acg_minimum_point = 2000;
    % ccg_threshold = 0.25;
    acg_threshold = 0.25;  % acg means crosscorrelogram
    % specificity_thres = 0.5; % specificity will be 0.5 if there are from the same distribution

    n1 = size(pc_scores_cluster1, 1);
    n2 = size(pc_scores_cluster2, 1);
    % [R, ~] = corr(X'); I had a problem for bimodal projection, as the
    % first clusters is big (messy), the projection is not centered at -1
    % but centered at 0. Therefore, although the two clusters should not be
    % merged, the bimodal score will be low. I tried the following
    % approach, but it does not solve the problem instead of using weighted
    % least square, we can oversample empirically, we observe Determine the
    % smaller cluster and its target size for oversampling Assume
    % pc_scores_cluster1 and pc_scores_cluster2 are defined somewhere in
    % your script
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


    new_samples = mvnrnd(pc_mean, pc_cov, num_new_samples);
    if cluster_to_oversample == 1
        pc_scores_cluster1_resampled = [pc_scores_cluster1; new_samples];
        X = [pc_scores_cluster1_resampled; pc_scores_cluster2];
        y = [ones(size(pc_scores_cluster1_resampled, 1), 1); 2 * ones(size(pc_scores_cluster2, 1), 1)];
    else
        pc_scores_cluster2_resampled = [pc_scores_cluster2; new_samples];
        X = [pc_scores_cluster1; pc_scores_cluster2_resampled];
        y = [ones(size(pc_scores_cluster1, 1), 1); 2 * ones(size(pc_scores_cluster2_resampled, 1), 1)];
    end

    % % viz to check oversampling procedure - checked
    % figure;
    % hold on;
    % plot((pc_scores_cluster1  * retained_ceoff')' + mu', 'Color', [1 0 0 0.1]);
    % plot((pc_scores_cluster2_resampled * retained_ceoff')' + mu', 'Color', [0 0 1 0.1]);
    % hold off;
    
    % % second implementation 
    % X = [pc_scores_cluster1; pc_scores_cluster2];
    % y = [ones(size(pc_scores_cluster1, 1), 1); 2*ones(size(pc_scores_cluster2, 1), 1)]; % Labels
    % [idx, ~] = kmeans(X, 2);
    % cm1 = confusionmat(y, idx); 
    % % in cm1 we have col as new cluster 1 and 2 
    % % row as olf cluster 1 and 2, cm(1,1) is the number of samples from old
    % % cluster 1 is classified as new cluster 1, and cm (2,1) is the number
    % % of samples from old cluster 2 is classfied as new cluster 2
    % cm2 = confusionmat(y, 3 - idx); 
    % specificity_score1 = (cm1(1, 1) / sum(cm1(:, 1)) + cm1(2, 2) / sum(cm1(:, 2)))/ 2;
    % specificity_score2 = (cm2(1, 1) / sum(cm2(:, 1)) + cm2(2, 2) / sum(cm2(:, 2)))/ 2;
    % specificity_score = max(specificity_score1, specificity_score2);
    % 
    % % permutation to get threshold
    % permuted_specificity_scores = zeros(num_permutations, 1);
    % idx_ones = find(y == 1);
    % idx_twos = find(y == 2);
    % for i_perm = 1:num_permutations
    %     flip_twos = randsample(idx_twos, max(1, floor(length(idx_twos) *...
    %         permutation_proportion)));
    %     flip_ones = randsample(idx_ones, max(1, floor(length(idx_ones) *...
    %         permutation_proportion)));
    %     y_permuted = y;
    %     y_permuted(flip_twos) = 1;
    %     y_permuted(flip_ones) = 2;
    %     cm1_perm = confusionmat(y_permuted, idx);
    %     cm2_perm = confusionmat(y_permuted, 3 - idx);
    %     specificity_score1_perm = (cm1_perm(1, 1) / sum(cm1_perm(:, 1)) + cm1_perm(2, 2) / sum(cm1_perm(:, 2))) / 2;
    %     specificity_score2_perm = (cm2_perm(1, 1) / sum(cm2_perm(:, 1)) + cm2_perm(2, 2) / sum(cm2_perm(:, 2))) / 2;
    %     permuted_specificity_scores(i_perm) = max(specificity_score1_perm, specificity_score2_perm);
    % end
    % specificity_thres = prctile(permuted_specificity_scores, merge_prctil);

    % stability using Adjusted Rand Index (ARI)
    stability_scores = zeros(num_permutations, 1); 
    stability_scores_perm = zeros(num_permutations, 1); 
    
    for i = 1:num_permutations
        resample_indices = randsample(1:n1+n2, n1+n2, true);
        X_resample = X(resample_indices, :);
        idx_resample = y(resample_indices);
        idx_recluster = kmeans(X_resample, 2);
        stability_scores(i) = rand_index(idx_recluster, idx_resample, 'adjusted');
        % perm
        idx_ones = find(idx_recluster == 1);
        idx_twos = find(idx_recluster == 2);
        flip_twos = randsample(idx_twos, max(1, floor(length(idx_twos) *...
            permutation_proportion)));
        flip_ones = randsample(idx_ones, max(1, floor(length(idx_ones) *...
            permutation_proportion)));
        idx_recluster_perm = idx_recluster;
        idx_recluster_perm(flip_twos) = 1;
        idx_recluster_perm(flip_ones) = 2;
        stability_scores_perm(i) = rand_index(idx_resample, ...
            idx_recluster_perm, 'adjusted');
    end
    avg_stability = mean(stability_scores);
    stability_thres = prctile(stability_scores_perm, merge_prctil);
    
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
        if (avg_stability < stability_thres)
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
            if (avg_stability < stability_thres)
                shouldMerge = true;
            % we pass acg test, then we add a soft bimod_thres
            else
                shouldMerge = false;
            end
        end
    end

end
