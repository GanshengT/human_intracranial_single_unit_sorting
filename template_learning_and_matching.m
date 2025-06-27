function [spike_2_export] = template_learning_and_matching(recording, sfreq, ...
    save_dir, varargin)
    %% description
    % This function is used to detect neurons for each channel
    % INPUT:
    % signal: SUA recording for one Behnke Fried electrode (time * ch);
    % note that this function does not support single channel sorting.
    % sfreq: sampling rate
    % save_dir: output directory
    % varargin{1}: chosen interval to display processing steps
    % varargin{2}: firing_rate_criterion, ignore channels that has few
    % spikes (the detected spikes has firing rate less than firing_rate_criterion)

    % OUPUT:
    % visualization of time series is on ms
    
    % Author: Gansheng Tan (g.tan@wustl.edu)

    % add path for graph theory
    community_detction = 'ComDetTBv090';
    addpath(community_detction)
    addpath([community_detction '/Algorithms'])
    addpath([community_detction '/Auxiliary'])
    addpath([community_detction '/Cluster_Number'])
    addpath([community_detction '/ClusterValidity'])
    addpath([community_detction '/Evaluation'])
    addpath([community_detction '/Experiments'])
    addpath([community_detction '/Graphs'])
    addpath([community_detction '/Help'])
    
    %% predefined parameters
    spike_low_cutoff = 300; 
    spike_high_cutoff = 6000; 
    filter_order = 4;
    epsilon = 1e-5; 
    batch_overlap_time = 0.2; % 0.2s
    multiplier_detection_threshold = 5; 
    batch_time = 5;
    pre_interval_spike = 1e-3; % 1ms
    post_interval_spike = 1e-3; %1ms
    % PCA parameters
    min_num_pcs = 4;
    max_num_pcs = 7;
    variance_require_stop_pca = 0.9;
    increment_variance_stop_pca = 0.03;
    % community algorithm parameter
    batch_size_graph = 10000;
    community_size_requirement_for_spike = 0.01; 
    % community with size less than 1% will be pruned
    % oversplit-and-merge parameters
    num_initial_clusters = 3; % additional clusters for oversplit
    cluster_minimum_factor = 8;
    max_n_peak_spike_shape = 3;
    min_n_peak_spike_shape = 1;
    merge_prctil = 97;
    if nargin < 4
        chosen_interval = 5e3:6e3;
    else
        chosen_interval = varargin{1};
    end

    if nargin < 5
        firing_rate_criterion = 0.2;
    else
        firing_rate_criterion = varargin{2};
    end

    if nargin < 6
        merge_prctil = 97;
    else
        merge_prctil = varargin{3};
    end

    % template matching parameters
    batch_size_optim = 0.4; % 0.4s
    max_firing_rate = 100; % firing rate can not be larger than 100Hz
    % output summary parameters
    window_length_firing_rate = 2; %2s
    step_size_firing_rate = 1; %1s

    %% preprocessing
    % prep - common reference only if we have more than 1 ch
    if size(recording, 2) > 1
        ch_mean = mean(recording, 1);
        signal_mic_common_referenced = recording - ch_mean;
        common_average = mean(signal_mic_common_referenced, 2);
        signal_mic_common_referenced = ...
            bsxfun(@minus, double(signal_mic_common_referenced), common_average);
    else
        signal_mic_common_referenced = recording;
    end
    
    % prep - filtering
    Wn = [spike_low_cutoff spike_high_cutoff]/ (sfreq / 2);
    [b, a] = butter(filter_order, Wn, 'bandpass');
    signal_mic_bandpassed = filtfilt(b, a, signal_mic_common_referenced);
    
    % visualization of processing
    figure('Position', [100 100 3 * 400 size(recording, 2) * 300]);
    
    for i = 1:size(recording, 2)
        subplot(size(recording, 2), 3, i * 3 - 2);
        plot(chosen_interval / sfreq * 1e3, recording(chosen_interval, i));
        xlabel('Time (ms)');
        ylabel('Amplitude (uV)');
        title(['raw ch', num2str(i)]);
        subplot(size(recording, 2), 3, i * 3 - 1);
        plot(chosen_interval / sfreq * 1e3, ...
            signal_mic_common_referenced(chosen_interval, i));
        title(['referenced ch', num2str(i)]);
        subplot(size(recording, 2), 3, i * 3);
        plot(chosen_interval / sfreq * 1e3, ...
            signal_mic_bandpassed(chosen_interval, i));
        title(['filtered ch', num2str(i)]);
    end
    dateTimeNow = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = fullfile(save_dir, ['signal_processing_raw_reref_filtered_' ...
        dateTimeNow '.svg']);
    saveas(gcf, filename, 'svg');
    
    % prep - channel whitening
    % channels whitening
    covariance_matrix = cov(signal_mic_bandpassed);
    [U, S, V] = svd(covariance_matrix);
    S_epsilon_smoothed = S + epsilon * eye(size(S));
    
    S_inv_sqrt = diag(1 ./ sqrt(diag(S_epsilon_smoothed)));
    W = U * S_inv_sqrt * U';
    signal_whitened = W * signal_mic_bandpassed';
    signal_whitened = signal_whitened';
    
    %% rough spike detection
    % detection by patch and get the template pool
    pre_interval_spike_sample = sfreq * pre_interval_spike; % 1ms
    post_interval_spike_sample = sfreq * post_interval_spike; % 1ms
    pre_interval_spike_sample = 2 ^ nextpow2(pre_interval_spike_sample);
    post_interval_spike_sample = 2 ^ nextpow2(post_interval_spike_sample);
    batch_size = sfreq * batch_time; 
    batch_overlap_size = sfreq * batch_overlap_time;
    num_samples = size(signal_whitened, 1);
    num_batches = ceil((num_samples - batch_overlap_size) / (batch_size - batch_overlap_size));

    spike_times = cell(size(signal_whitened, 2), num_batches);
    spike_shapes = cell(size(signal_whitened, 2), num_batches);
    
    for i_ch = 1:size(signal_whitened, 2)
        current_ch_signal = signal_whitened(:, i_ch);
        for b = 1:num_batches
            batch_start = (b - 1) * (batch_size - batch_overlap_size) + 1;
            batch_end = min(batch_start + batch_size - 1, length(current_ch_signal));
            batch_data = current_ch_signal(batch_start:batch_end); 
            % similar to osort 
            sigma_n = median(abs(batch_data)) / 0.6745;
            threshold = multiplier_detection_threshold * sigma_n;
            spike_indices = find(abs(batch_data) > threshold);
    
            % Reduce continuous spike indices， we are forming templates, it's
            % fine to omit one or two spikes
            diff_spike_indices = [diff(spike_indices); 1]; 
            spike_groups = find(diff_spike_indices > (pre_interval_spike_sample + ...
                post_interval_spike_sample) );
            reduced_spike_indices = zeros(length(spike_groups), 1);
            start_idx = 1;
            for i = 1:length(spike_groups)
                end_idx = spike_groups(i);
                reduced_spike_indices(i) = round(mean(spike_indices(start_idx:end_idx)));
                start_idx = end_idx + 1;
            end
            spike_indices = reduced_spike_indices;
            % test
            % figure;
            % hold on
            for i = 1:length(spike_indices)
                local_max_range = max(1, spike_indices(i) - pre_interval_spike_sample):...
                min(length(batch_data), spike_indices(i)  + post_interval_spike_sample);
                [~, local_max_idx] = max(abs(batch_data(local_max_range)));
                aligned_spike_idx = local_max_range(1) + local_max_idx - 1;

                if (aligned_spike_idx > pre_interval_spike_sample) && ( ...
                        (length(batch_data) - aligned_spike_idx) > post_interval_spike_sample)
                    spike_times{i_ch, b} = [spike_times{i_ch, b}; aligned_spike_idx + batch_start - 1];
                    spike_shapes{i_ch, b}{end+1} = batch_data(aligned_spike_idx - pre_interval_spike_sample: ...
                        aligned_spike_idx + post_interval_spike_sample);
                    % test - % make sure no overlap at this stage - it will
                    % create alignment issue
                    % if max(abs(batch_data(aligned_spike_idx - pre_interval_spike_sample: ...
                    %     aligned_spike_idx + post_interval_spike_sample))) ~= abs(spike_shapes{i_ch, b}{end}(33))
                    %     disp(num2str(i))
                    % end
                    % plot(abs(batch_data(aligned_spike_idx - pre_interval_spike_sample: ...
                    %     aligned_spike_idx + post_interval_spike_sample)))
                end
            end
        end
    end
    
    unique_spike_shapes = cell(size(signal_whitened, 2), 1);
    unique_spike_times = cell(size(signal_whitened, 2), 1);
    
    for i_ch = 1:size(signal_whitened, 2)
        combined_spike_times = sort(vertcat(spike_times{i_ch, :}));
        % Remove duplicates: spikes that occur within overlap_size of each other
        % are considered duplicates - only the first occurrence is kept
        diff_spike_times = [Inf; diff(combined_spike_times)];
        non_duplicate_indices = diff_spike_times > 0; 
        unique_spike_times{i_ch} = combined_spike_times(non_duplicate_indices); 
        combined_spike_shapes = {};
        for b = 1:num_batches
            combined_spike_shapes = [combined_spike_shapes, spike_shapes{i_ch, b}];
        end
        unique_spike_shapes{i_ch} = combined_spike_shapes(non_duplicate_indices);
    end

    figure('Position', [100 100 3 * 400 size(recording, 2) * 300]);
    for i = 1:size(recording, 2)
        subplot(size(recording, 2), 3, i * 3 - 2);
        plot(chosen_interval / sfreq * 1e3, ...
            signal_mic_bandpassed(chosen_interval, i));
        title(['filtered ch', num2str(i)]);
        subplot(size(recording, 2), 3, i * 3 - 1);
        plot(chosen_interval / sfreq * 1e3, ...
            signal_whitened(chosen_interval, i));
        title(['whitened ch', num2str(i)]);
        subplot(size(recording, 2), 3, i * 3);
        plot(chosen_interval / sfreq * 1e3, signal_whitened(chosen_interval, i), 'b');
        hold on;
        spike_times_in_interval = unique_spike_times{i} - chosen_interval(1) + 1;
        spike_times_in_interval = spike_times_in_interval(spike_times_in_interval >= 1 &...
            spike_times_in_interval <= length(chosen_interval));

        for j = 1:length(spike_times_in_interval)
            time_interval = [spike_times_in_interval(j) - pre_interval_spike_sample,...
                spike_times_in_interval(j) + post_interval_spike_sample];
            spike_interval = chosen_interval(time_interval(1):time_interval(2));
            plot(spike_interval / sfreq * 1e3, signal_whitened(spike_interval, i), 'r');
        end
        hold off;
        title(['Detected Spikes ch', num2str(i)]);
    end
    filename = fullfile(save_dir, ['signal_processing_filtered_whitened_detected_spk' ...
        dateTimeNow '.svg']);
    saveas(gcf, filename, 'svg');

    % variable management
    clear signal_mic_common_referenced
    clear signal_mic_bandpassed
    clear spike_interval
    clear current_ch_signal
    clear S
    clear U
    clear V
    clear S_inv_sqrt
    clear W
    clear non_duplicate_indices
    clear S_epsilon_smoothed
    clear common_average
    clear diff_spike_times
    clear diff_spike_indices
    clear spike_groups

    %% template learning
    expected_spike_num = num_samples / ...
        (1 / firing_rate_criterion * sfreq);
    for i_ch = 1:length(unique_spike_shapes)
        if length(unique_spike_shapes{i_ch}) < expected_spike_num
            disp(['ignore ch ' num2str(i_ch), 'because spikes detected are not ' ...
                'enough for template learning']);
            continue
        end

        all_spike_shapes = unique_spike_shapes(i_ch);
        all_spike_shapes = [all_spike_shapes{:}];
        all_spike_shapes = cell2mat(all_spike_shapes);
        [coeff, score, ~, ~, explained, mu] = pca(all_spike_shapes');
        cumulativeVariance = cumsum(explained);
        num_pcs_retained = find(cumulativeVariance >= (variance_require_stop_pca * 100), 1, 'first');
        num_pcs_retained = min([find(explained <= (increment_variance_stop_pca * 100), 1, 'first'), ...
            num_pcs_retained, max_num_pcs]);
        if num_pcs_retained < min_num_pcs
            num_pcs_retained = min_num_pcs;
        end
        % if there is only one cluster, the explained variance will be
        % small, do not worry too much
        retained_coeff = coeff(:, 1:num_pcs_retained);
        retained_scores = score(:, 1:num_pcs_retained);

        % verification of PCA
        figure; 
        subplot(2, 1, 1)
        plot(all_spike_shapes(:, 5))
        hold on;
        plot((retained_scores(5,:) * retained_coeff')' + mu');
        legend('Original Spike', 'Reconstructed Spike');
        xlabel('Sample Index');
        ylabel('Amplitude');
        title('Comparison of Original and Reconstructed Spike');
        subplot(2, 1, 2)
        plot(all_spike_shapes(:, 6))
        hold on;
        plot((retained_scores(6,:) * retained_coeff')' + mu');
        legend('Original Spike', 'Reconstructed Spike');
        xlabel('Sample Index');
        ylabel('Amplitude');
        title('Comparison of Original and Reconstructed Spike');
        filename = fullfile(save_dir, ['ch_' num2str(i_ch) '_signal_processing_reconstructed_spk_pca' ...
        dateTimeNow '.svg']);
        saveas(gcf, filename, 'svg');

        num_batches = max(1, floor(size(retained_scores, 1) / batch_size_graph));
        batch_size = ceil(size(retained_scores, 1) / num_batches);
        toKeep = true(size(retained_scores, 1), 1);
        figure('Position', [100 100 300 * 5 num_batches * 300]);
        n_spike_com = [];
        for b = 1:num_batches
            start_idx = (b - 1) * batch_size + 1;
            end_idx = min(b * batch_size, size(retained_scores, 1));
            pairwiseDistances = pdist2(retained_scores(start_idx:end_idx, :),...
                retained_scores(start_idx:end_idx, :));
            A_weighted = exp(-pairwiseDistances);
            upperTriA = triu(A_weighted, 1);
            edgeWeights = upperTriA(upperTriA > 0);
            sortedEdgeWeights = sort(edgeWeights, 'ascend');
            edgeIndices = 1:length(sortedEdgeWeights); 
            firstPoint = [edgeIndices(1), sortedEdgeWeights(1)];
            lastPoint = [edgeIndices(end), sortedEdgeWeights(end)];
            lineVec = lastPoint - firstPoint;
            lineVecN = lineVec / norm(lineVec);
            vecFromFirst = bsxfun(@minus, [edgeIndices', sortedEdgeWeights], firstPoint);
            scalarProduct = dot(vecFromFirst, repmat(lineVecN, numel(sortedEdgeWeights), 1), 2);
            vecFromFirstParallel = scalarProduct * lineVecN;
            vecToLine = vecFromFirst - vecFromFirstParallel;
            distToLine = sqrt(sum(vecToLine.^2, 2));
            [~, idx] = max(distToLine); 
            kneeThreshold = sortedEdgeWeights(idx); 
            subplot(num_batches, 5, 5*b-4)
            plot(sortedEdgeWeights, 'b.-');
            title('Sorted Edge Weights');
            xlabel('Edge Index');
            ylabel('Edge Weight');
            hold on;
            plot(idx, sortedEdgeWeights(idx), 'ro', 'MarkerFaceColor', 'r');
            legend('Sorted Edge Weights', 'Elbow Point');

            A_weighted(A_weighted < kneeThreshold) = 0;
            A_weighted(logical(eye(size(A_weighted)))) = 0;
            community_label = GCModulMax1(A_weighted);

            minimum_occurrence = max(round(size(pairwiseDistances, 1) * ...
                community_size_requirement_for_spike), 2);
            unique_communities = unique(community_label);
            for i_com = 1:length(unique_communities)
                current_community = unique_communities(i_com);
                members = find(community_label == current_community);
                if numel(members) < minimum_occurrence
                    toKeep(members + start_idx - 1) = false;
                end
            end
            % community_labels_keep = community_label(toKeep(start_idx:end_idx));
            [communityCounts, uniqueCommunityLabels] = groupcounts(community_label);
            [com_counts, com_idx] = sort(communityCounts, 'descend');
            n_spike_com(end + 1) = sum(com_counts >= minimum_occurrence);
            pruned_com = com_idx(com_counts < minimum_occurrence);
            selected_communities = randsample(pruned_com, 2);
            for i_community = 1:2
                current_com_label = uniqueCommunityLabels(com_idx(i_community));
                current_indices = (community_label == current_com_label);
                spike_indices = zeros(size(all_spike_shapes, 2), 1);
                spike_indices(start_idx:end_idx) = current_indices;
                spikeShapesLargest = all_spike_shapes(:, spike_indices==1);
                subplot(num_batches, 5, 5*b-4 + i_community)
                plot(spikeShapesLargest, 'Color', [0 0 1 0.1]);
                title([num2str(i_community) 'largest Community (' ...
                    num2str(com_counts(i_community)) ')']);
                xlabel('Sample Index');
                ylabel('Amplitude');
                averageSpikeShape = mean(spikeShapesLargest, 2);
                hold on;
                plot(averageSpikeShape, 'LineWidth', 2, 'Color', 'r');
                xlim([0 size(all_spike_shapes,  1)])

                current_com_label = selected_communities(i_community);
                current_indices = (community_label == current_com_label);
                spike_indices = zeros(size(all_spike_shapes, 2), 1);
                spike_indices(start_idx:end_idx) = current_indices;
                spikeShapesLargest = all_spike_shapes(:, spike_indices==1);
                subplot(num_batches, 5, 5*b-2 + i_community)
                plot(spikeShapesLargest, 'Color', [0 0 1 0.1]);
                title([num2str(i_community) 'pruned Community (' ...
                    num2str(com_counts(current_com_label)) ')']);
                xlabel('Sample Index');
                ylabel('Amplitude');
                averageSpikeShape = mean(spikeShapesLargest, 2);
                hold on;
                plot(averageSpikeShape, 'LineWidth', 2, 'Color', 'r');
                xlim([0 size(all_spike_shapes,  1)])
            end    
        end
        filename = fullfile(save_dir, ['ch_' num2str(i_ch) 'spike and noise community' ...
        dateTimeNow '.svg']);
        saveas(gcf, filename, 'svg');
        disp([num2str(sum(toKeep)) ' out of ' num2str(length(toKeep)) ...
            ' detected shapes are classified as spikes']);
        % remove detected noise
        all_spike_shapes = all_spike_shapes(:, toKeep);
        all_spike_times = unique_spike_times(i_ch);
        all_spike_times = all_spike_times{1};
        all_spike_times = all_spike_times(toKeep);
        %% template-learning: oversplit - merge
        [coeff, score, ~, ~, explained, mu] = pca(all_spike_shapes');
        cumulativeVariance = cumsum(explained);
        num_pcs_retained = find(cumulativeVariance >= (variance_require_stop_pca * 100), 1, 'first');
        num_pcs_retained = min([find(explained <= (increment_variance_stop_pca * 100), 1, 'first'), ...
            num_pcs_retained, max_num_pcs]);
        if num_pcs_retained < min_num_pcs
            num_pcs_retained = min_num_pcs;
        end
        retained_coeff = coeff(:, 1:num_pcs_retained);
        retained_scores = score(:, 1:num_pcs_retained);

        % verification of PCA
        figure; 
        subplot(2, 1, 1)
        plot(all_spike_shapes(:, 5))
        hold on;
        plot((retained_scores(5,:) * retained_coeff')' + mu');
        legend('Original Spike', 'Reconstructed Spike');
        xlabel('Sample Index');
        ylabel('Amplitude');
        title('Comparison of Original and Reconstructed Spike');
        subplot(2, 1, 2)
        plot(all_spike_shapes(:, 6))
        hold on;
        plot((retained_scores(6,:) * retained_coeff')'+ mu');
        legend('Original Spike', 'Reconstructed Spike');
        xlabel('Sample Index');
        ylabel('Amplitude');
        title(['PCA explains' num2str(cumulativeVariance(num_pcs_retained)) '% variance']);
        filename = fullfile(save_dir, ['ch_' num2str(i_ch) '_reconstructed_spk_pca_after_removing_noise' ...
        dateTimeNow '.svg']);
        saveas(gcf, filename, 'svg');

        % parameters management
        clear com_idx
        clear A_weighted
        clear batch_data
        clear com_counts
        clear communityCounts
        clear community_labels_pruned
        clear community_labels_keep
        clear current_indices
        clear edgeIndices
        clear edgeWeights
        clear indicesLargest
        clear pairwiseDistances
        clear pruned_com
        clear scalarProduct
        clear vecFromFirstParallel
        clear vecFromFirst
        clear vecToLine
        clear distToLine
        clear upperTriA

        % [idx, C] = kmeans(retained_scores, num_initial_clusters, ...
        %     'Distance', 'sqeuclidean', 'Replicates', 10, 'Start', 'plus');
        [idx, C] = kmeans(retained_scores, max(n_spike_com) + num_initial_clusters, ...
            'Distance', 'sqeuclidean', 'Replicates', 10, 'Start', 'plus');
        shouldContinue = true;
        while shouldContinue
            cos_sim_centers = pdist2(C, C, 'euclidean');
            cos_sim_centers(tril(true(size(cos_sim_centers)))) = NaN; 
            cos_sim_centers = max(cos_sim_centers) -  cos_sim_centers;
            % get the indices of the most similar pair
            [sortedSims, sortIdx] = sort(cos_sim_centers(:), 'descend');
            validPairs = sortIdx(sortedSims >= 0); % Only consider similarities > 0
            if isempty(validPairs)
                shouldContinue = false; 
                break
            end
            for i_pair = 1:length(validPairs)
                [row, col] = ind2sub(size(cos_sim_centers), validPairs(i_pair));
                cluster_1_id = (idx == row);
                cluster_2_id = (idx == col);
                pc_scores_cluster1 = retained_scores(cluster_1_id, :);
                pc_scores_cluster2 = retained_scores(cluster_2_id, :);
                % ignore small clusters, they will be classified as noise
                % template later
                if (size(pc_scores_cluster2, 1) < cluster_minimum_factor * size(C, 2)) ||...
                        (size(pc_scores_cluster1, 1) < cluster_minimum_factor * size(C, 2))
                    continue
                end
                all_spike_times_cluster1 = all_spike_times(cluster_1_id) /...
                sfreq;
                all_spike_times_cluster2 = all_spike_times(cluster_2_id) /...
                    sfreq;       
                % viz - for debugging purpose
                % figure;
                % hold on;
                % plot((pc_scores_cluster1 * retained_coeff')' + mu', 'b');
                % plot((pc_scores_cluster2 * retained_coeff')' + mu', 'r');
                % hold off;
                % title('reconstructed spikes')
                % figure;
                % corresponding_spike_shapes_cluster1 = all_spike_shapes(:, cluster_1_id);
                % corresponding_spike_shapes_cluster2 = all_spike_shapes(:, cluster_2_id);
                % hold on;
                % plot(corresponding_spike_shapes_cluster1, 'Color',[0,0,1,0.5]);
                % plot(corresponding_spike_shapes_cluster2, 'Color',[1,0,0,0.5]);
                % hold off;

                % pass the retained_coeff in the function for debugging
                % convenience
                if decideMerge_bci2000(pc_scores_cluster1, ...
                        pc_scores_cluster2, all_spike_times_cluster1, ...
                        all_spike_times_cluster2, merge_prctil, retained_coeff, mu)
                    % do merge, update C and idx
                    idx(idx == col) = row;
                    C(row, :) = mean(retained_scores(idx == row, :), 1);
                    % renumbering idx and C, because we remove one component
                    C(col, :) = []; % This removes the 'col'th row from C
                    for i = col:max(idx)-1
                        idx(idx == i + 1) = i;
                    end
                    break;
                end
            end
            if i_pair == length(validPairs)
                shouldContinue = false; 
            end
        end
        % alternative approach will be using reconstructed template
        % (retained_coeff * C(i_template, :)')'
        systematic_noise_template = [];
        num_per_row = 4; 
        num_rows = ceil(size(C, 1) / num_per_row);
        y_lim_4_figure = zeros(2, size(C, 1));
        for i_template = 1:size(C, 1)
            current_spike_template = (retained_coeff * C(i_template, :)')' + mu;
            y_lim_4_figure(2, i_template) = max(current_spike_template);
            y_lim_4_figure(1, i_template) = min(current_spike_template);
        end

        figure;
        for i_template = 1:size(C, 1)
            current_spike_template = (retained_coeff * C(i_template, :)')' + mu;
            current_spike_template_norm = (current_spike_template - ...
                mean(current_spike_template)) / std(current_spike_template);
            prominant_peak_thres = 0.5 * (max(abs(current_spike_template_norm)));
            n_prominant_peak = length(findpeaks(abs(current_spike_template_norm), ...
                'MinPeakProminence', prominant_peak_thres));
            % figure;
            % plot(current_spike_template);
            % hold on;
            % % yline(prominant_peak_thres);
            % title(['template' num2str(i_template)]);
            % disp(['template' num2str(i_template) ': ', num2str(n_prominant_peak)])
            if (sum(idx==i_template) < (num_samples / ...
                (1 / firing_rate_criterion * sfreq))) ||...
                n_prominant_peak > max_n_peak_spike_shape ||...
                n_prominant_peak < min_n_peak_spike_shape
                systematic_noise_template(end + 1) = i_template;
                noiseOrSpike = 'Noise';
            else
                noiseOrSpike = 'Spike';
            end
            corresponding_spike_indices = (idx == i_template);
            corresponding_spike_shapes = all_spike_shapes(:, corresponding_spike_indices);
            subplot(num_rows, num_per_row, i_template);
            hold on;
            plot(corresponding_spike_shapes, 'Color', [0, 0, 1, 0.1]);
            plot(current_spike_template, 'r', 'LineWidth', 2); 
            titleStr = sprintf('T %d: %d peaks, %s', ...
                i_template, n_prominant_peak, noiseOrSpike);
            title(titleStr);
            hold off;
            xlabel('Sample Index');
            ylabel('Amplitude');
            ylim([min(y_lim_4_figure(1, :)) - 1, max(y_lim_4_figure(2, :)) + 1])
        end
        filename = fullfile(save_dir, ['ch_' num2str(i_ch) '_templates_for_deconvlt' ...
        dateTimeNow '.svg']);
        saveas(gcf, filename, 'svg');
        keepTemplates = true(size(C, 1), 1);
        keepTemplates(systematic_noise_template) = false;
        C_spike = C(keepTemplates, :);
        %% template matching - optimization
        signal_current_ch = signal_whitened(:, i_ch);
        % take a batch to calculate lambda_max
        batch_size = sfreq * batch_size_optim;
        template_length = size(retained_coeff, 1);
        step = batch_size - template_length;
        refractory_period = 2 ^ nextpow2(pre_interval_spike_sample) + 1;
        solutions = zeros(length(signal_current_ch), size(C_spike, 1));
        startIdx = 1;
        A_combined = [];
        for i_template = 1:size(C_spike, 1)
            current_spike_template = C_spike(i_template, :) * ...
                retained_coeff' + mu;
            A = createToeplitz_spike_template(current_spike_template', batch_size);
            A_combined = [A_combined, A]; 
        end
        while startIdx < length(signal_current_ch) 
            endIdx = startIdx + batch_size - 1;
            if endIdx > length(signal_current_ch)
                break
            end
            currentBatch = signal_current_ch(startIdx:endIdx);
            % use lambda to penalize the number of 1s in the solution
            % vector
            % round(batch_size_optim / (1/max_firing_rate)) represents the
            % maximum number of spikes per neurons
            lambda_percentile = round(batch_size_optim / (1/max_firing_rate)) * size(C_spike, 1);
            % the decrease of objective function is we add a one in the
            % solution vector
            expectation_decrease_objective_fn = -1/2*sum(A_combined.^2, 1) + currentBatch'*A_combined;
            % [sortedValues, sortedIndices] = sort(expectation_decrease_objective_fn, 'descend');
            if max(expectation_decrease_objective_fn, [], 'omitnan') > 0
                % init xOpt as the full solution vector
                xOpt = zeros(size(A_combined, 2), 1);
                xOpt(expectation_decrease_objective_fn > 0) = 1;
                solution = reshape(xOpt, [endIdx-startIdx+1, size(C_spike, 1)]);
                expectation_decrease_objective_fn_reshape = ...
                    reshape(expectation_decrease_objective_fn, [endIdx-startIdx+1, ...
                    size(C_spike, 1)]);
                xOpt = applyRefractoryConstraint(solution, ...
                    expectation_decrease_objective_fn_reshape, refractory_period);
                [sortedValues, sortedIndices] = sort(expectation_decrease_objective_fn(xOpt==1), 'descend');
                if sum(xOpt==1) < lambda_percentile
                    lambda = min(sortedValues);
                else
                    % calculate lambda to restrict the firing rate, only
                    % consider the decrease of objective function for 1
                    % addition where xOpt==1 (objective function will
                    % decrease if adding 1 around the spike)
                    lambda = max(0, sortedValues(lambda_percentile));
                end

                xOpt = full_to_best_deconvolution(currentBatch, A_combined, xOpt', ...
                    lambda, size(retained_coeff, 1));
                solution = reshape(xOpt, [endIdx-startIdx+1, size(C_spike, 1)]);
            else
                % no need to optimize, no spikes
                solution = zeros(endIdx-startIdx+1, size(C_spike, 1));
            end
            
            % verify solution (A_combined * xOpt';) - passed
            solutions(startIdx:(endIdx - template_length), :) = solution(1:...
                (endIdx - startIdx - template_length + 1), :);
            startIdx = startIdx + step;
        end
        % store sorting results for each channel
        spike_2_export = struct('individual_spike_shapes', cell(1, size(C_spike, 1)), ...
           'spike_template', cell(1, size(C_spike, 1)), ...
           'spike_time', cell(1, size(C_spike, 1)), ...
           'firingRate_2s_window_1s_step', cell(1, size(C_spike, 1)),...
           'sfreq', sfreq);

        for i_template = 1:size(C_spike, 1)
            current_spike_template = C_spike(i_template, :) * ...
                retained_coeff' + mu;
            spike_2_export(i_template).spike_template = current_spike_template;
            spike_2_export(i_template).spike_time = find(solutions(:, i_template) == 1);

            firing_rate = []; 
            for startSample = 1:round(step_size_firing_rate * sfreq):(size(solutions, 1) -...
                    (window_length_firing_rate * sfreq)+ 1)
                endSample = startSample + window_length_firing_rate * sfreq - 1;
                spikesCount = sum((startSample <= spike_2_export(i_template).spike_time) ...
                    & (spike_2_export(i_template).spike_time <= endSample));
                firing_rate(end + 1) = spikesCount / window_length_firing_rate; 
            end
            spike_2_export(i_template).firingRate_2s_window_1s_step = firing_rate;

            if length(spike_2_export(i_template).spike_time) > 1 
                isi = diff(spike_2_export(i_template).spike_time) / sfreq; 
                spike_2_export(i_template).ISI = isi;
            else
                spike_2_export(i_template).ISI = []; 
            end
            individualSpikeShapes = zeros(length(current_spike_template), ...
                length(spike_2_export(i_template).spike_time)); 
            
            for i_spike = 1:length(spike_2_export(i_template).spike_time)
                start_spike = spike_2_export(i_template).spike_time(i_spike);
                end_spike = start_spike + length(current_spike_template) - 1;
                
                if start_spike < 1
                    continue
                end
                if end_spike > length(signal_current_ch)
                    continue
                end
             
                spikeShape = signal_current_ch(start_spike:end_spike);
                individualSpikeShapes(:, i_spike) = spikeShape;
            end
            spike_2_export(i_template).individual_spike_shapes = individualSpikeShapes;

            % we will visualize found spikes with Gaussian smoothing
            % pdf: density of spike amplitudes across a
            % specified time window,

            figure('Position', [100, 100, 800, 600]);
            subplot(2, 3, 1);
            hold on;
            for i_spike = 1:size(individualSpikeShapes, 2)
                plot(individualSpikeShapes(:, i_spike), 'Color', [0 0 1 0.1]); 
            end
            SNR = mean(individualSpikeShapes(ceil(length(current_spike_template) / 2), ...
                :)) / std(individualSpikeShapes(ceil(length(current_spike_template) / 2), ...
                :));
            plot(current_spike_template, 'LineWidth', 2, 'Color', 'r');
            title(['Detected_spk SNR (meanPeak/std)=' SNR ')']);
            xlabel('Time Point');
            ylabel('Amplitude');
            allSpikesFlat = individualSpikeShapes(:);
            lowerBound = prctile(allSpikesFlat, 1);
            upperBound = prctile(allSpikesFlat, 99); 
            ylim([prctile(allSpikesFlat, 0.1) prctile(allSpikesFlat, 99.9)])
            hold off;
            
            % Subplot 2: Visualize the probability density of spikes
            [D, AMPs, bins] = individualSpikePDF(individualSpikeShapes); 
            subplot(2, 3, 2);
            hImg = imagesc(1:size(D, 2), AMPs, D);
            set(gca, 'YDir', 'normal'); 
            xlabel('Time Point');
            ylabel('Amplitude');
            title('Probability Density of Individual Spike Shapes');
            mask = D > ceil(size(individualSpikeShapes, 2) / bins / 10); 
            set(hImg, 'AlphaData', mask); 
            hold on
            plot(current_spike_template, '--', 'LineWidth', 2, 'Color', 'r');
            title('Probability Density of Spikes');
            hColorbar = colorbar;
            ylabel(hColorbar, {'Amplitude Occurrence Density within 1 out of ' + string(bins) + ' bins', ...
                   'between ' + string(min(allSpikesFlat)) + ' and ' + ...
                   string(max(allSpikesFlat))}); 

            ylim([prctile(allSpikesFlat, 0.1) prctile(allSpikesFlat, 99.9)])
            hold off
            
            isi = spike_2_export(i_template).ISI; 
            proportionUnder3ms = sum(isi < 0.003) / length(isi);
            
            if ~isempty(isi)
                subplot(2, 3, 3); 
                histogram(isi, 'Normalization', 'pdf'); 
                hold on;
                [density, xi] = ksdensity(isi); 
                plot(xi, density, 'r-', 'LineWidth', 2); % Plot the density estimation
                xlabel('Interspike Interval (s)');
                ylabel('Probability Density');
                title(['ISI Distribution mean = ', num2str(mean(isi))]);
                % legend('Histogram', 'Kernel Density Estimation');
                xlim([0, prctile(isi, 99)]);
                hold off;

                subplot(2, 3, 4);
                hold on
                edges = 0:1:200; 
                isi = isi((0 <= isi) & (isi < 0.2));
                histogram(isi*1000, 'BinEdges', edges, 'Normalization', 'pdf', 'EdgeColor', 'none');
                [density, xi] = ksdensity(isi*1000, 'Bandwidth', 0.1, 'Support', [0, 200]); 
                plot(xi, density, 'r-', 'LineWidth', 2); 
                xlim([0, 200]);
                xlabel('Interspike Interval (ms)');
                ylabel('Probability Density');
                title(['ISI <3ms: ' ...
                    num2str(proportionUnder3ms * 100, '%.2f') '%']);
                hold off
                legend('1ms Bins');
            else
                subplot(2, 3, 3); 
                title('No ISIs to Display');
                subplot(2, 3, 4); 
                title('No ISIs to Display');
            end

            subplot(2, 3, 5);
            left = -0.05 * 1e3;
            right = 0.05 * 1e3;
            bin = 0.0005 * 1e3;
            if ~isempty(spike_2_export(i_template).spike_time)
                spikeTimesInSeconds = spike_2_export(i_template).spike_time / sfreq * 1e3;
                diffST = spikeTimesInSeconds - spikeTimesInSeconds.';
                t = left:bin:right; 
                spikeTrain = histcounts(diffST, t); 
                spikeTrain(round((right - left) / bin / 2) + 1) = ...
                    spikeTrain(round((right - left) / bin / 2) + 1) - length(spikeTimesInSeconds);
                plot(t(1:end-1) + 0.001 * 1e3 / 2, spikeTrain);

                xlabel('Lag (ms)');
                ylabel('Autocorrelation');
                title('Spike Time Autocorrelation');
                line([0 0], ylim, 'Color', 'r', 'LineStyle', '--'); % Zero lag line
            else
                title('No Spike Times to Display');
            end


            subplot(2, 3, 6);
            timeVector = (1:length(spike_2_export(i_template).firingRate_2s_window_1s_step)) *...
                step_size_firing_rate; 
            windowSize = 10; 
            smoothedFiringRate = movmean(spike_2_export(i_template).firingRate_2s_window_1s_step,...
                windowSize);

            hold on;
            plot(timeVector, spike_2_export(i_template).firingRate_2s_window_1s_step, ...
                'b-', 'LineWidth', 1, 'DisplayName', 'Original Firing Rate');
            plot(timeVector, smoothedFiringRate, 'r-', 'LineWidth', 2, 'DisplayName', ...
                'Smoothed Firing Rate');
            hold off;
            xlabel('Time (s)');
            ylabel('Firing Rate (spikes/s)');
            title('Firing Rate (2s window)');
            legend('show'); 
            filename = fullfile(save_dir, ['ch_' num2str(i_ch) '_templates_' ...
                num2str(i_template) '_result_summary' ...
            dateTimeNow '.svg']);
            saveas(gcf, filename, 'svg');
        end

        % visualize signal and reconstructed spikes
        random_spike_time = spike_2_export(1).spike_time(...
            randi(length(spike_2_export(1).spike_time))); 
        segment_indices = (random_spike_time - (max(chosen_interval) - ...
            min(chosen_interval))):...
            (random_spike_time + (max(chosen_interval) - min(chosen_interval)));
        signal_segment = signal_current_ch(segment_indices);

        reconstructed_series = zeros(size(signal_segment));
        for i_template =1:size(spike_2_export, 2)
            current_spike_template = spike_2_export(i_template).spike_template;
            within_segment_indices = spike_2_export(i_template).spike_time >=...
                segment_indices(1) & spike_2_export(i_template).spike_time <= segment_indices(end);
            spike_times_within_segment = spike_2_export(...
                i_template).spike_time(within_segment_indices);
            for i_spike = 1:length(spike_times_within_segment)
                conv_v = zeros(size(signal_segment));
                if (spike_times_within_segment(i_spike) - ...
                    segment_indices(1) + ceil(length(current_spike_template) / 2)) <=...
                    size(signal_segment, 1)
                    conv_v(spike_times_within_segment(i_spike) - ...
                        segment_indices(1) + ceil(length(current_spike_template) / 2)) = 1;
                end
                reconstructed_series = conv(conv_v, current_spike_template, 'same') +...
                    reconstructed_series;
            end
        end


        figure; subplot(4,1,1);
        plot(segment_indices / sfreq, signal_segment, 'b'); hold on;
        plot(segment_indices / sfreq, reconstructed_series, 'r');
        xlabel('Time (s)'); ylabel('Amplitude');
        legend('Raw Signal', 'Reconstructed Spike Series');
        title('Signal and Reconstructed Spike Series');

        residuals = signal_segment - reconstructed_series;

        n = 64; 
        blueToWhite = [linspace(0, 1, n)' linspace(0, 1, n)' ones(n, 1)];
        whiteToRed = [ones(n, 1) linspace(1, 0, n)' linspace(1, 0, n)'];
        customCMap = [blueToWhite; whiteToRed];
        
        subplot(4,1,2);
        imagesc(signal_segment')
        colormap(customCMap);
        clim([-10, 10]);
        cb = colorbar(); 
        ylabel(cb,'Amplitude (a.u.)','FontSize',16)
        ylabel('Raw');
        xlabel('Sample index');


        subplot(4,1,3);
        imagesc(reconstructed_series')
        colormap(customCMap);
        clim([-10, 10]);
        cb = colorbar(); 
        ylabel(cb,'Amplitude (a.u.)','FontSize',16)
        ylabel('Reconstructed spikes');
        xlabel('Sample index');

        subplot(4,1,4);
        imagesc(residuals')
        colormap(customCMap);
        clim([-10, 10]);
        cb = colorbar(); 
        ylabel(cb,'Amplitude (a.u.)','FontSize',16)
        ylabel('Residual');
        xlabel('Sample index');

        filename = fullfile(save_dir, ['ch_' num2str(i_ch) '_raw_reconstruct_resid_' ...
        dateTimeNow '.svg']);
        saveas(gcf, filename, 'svg');

        % save results
        filename = fullfile(save_dir, ['ch_' num2str(i_ch) '_sorting_results_' ...
        dateTimeNow '.mat']);
        save(filename, 'spike_2_export', '-v7.3');

        % variable management
        close all;
        clear signal_segment
        clear segment_indices
        clear reconstructed_series
        clear residuals
        clear solutions

    end

end



