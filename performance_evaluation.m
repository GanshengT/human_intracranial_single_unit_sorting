%% description
% This script aims to test the performance of current template_matching
% method for spike sorting on bench-marking datasets
% The candidate datasets includde:
% 1. Visual coding data from Allen Brain Institute
% 2. Extracellular and intracellular recording in CRCNS - Collaborative Research in Computational 
% Neuroscience
% 3. SpikeForest
% In this script, we can use data from SpikeForest as other spike sorting
% algorithms have been tested on such data set.

% We will download the spike data and ground truth from spikeForest with
% the python interface. There are 5 datasets that are currently available:
% # Janelia for evaluation of drift correction, might be useful
% # synth monotrode data, low firing rate
% # boyden: intra and extra recording in mice
% # Kampff: also recording in mice https://www.biorxiv.org/content/10.1101/370080v2.full.pdf
% # english: awake behaving mice

% after download, the data will be saved in spikeForest_data and be organized as follows:
% 1. folder called 'study_set_name' correspond one dataset
% 2. In each dataset folder, the binary spike data along with parameters
% will be stored in a folder. The folder will be called
% {R.study_name}/{R.recording_name}
% 3. The unit (ground truth) is stored within each study folder. The file format is .mat

%% main
clear
dateTimeNow = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
% parameters
raw_filename = 'traces_cached_seg0.raw';
session_name = 'C_Easy1_noise01';
file_dir = ['spikeForest_data/spikeForest_data' ...
    '/SYNTH_MONOTRODE/neurocube_quiroga_easy1/'  ];
save_dir = 'spikeForest_output/';

% check if we can find raw_filename in this folder
full_file_path = fullfile(file_dir, session_name, session_name, raw_filename);
unit_info_file_path = fullfile(file_dir, session_name, [session_name '_true.mat']);

if exist(full_file_path, 'file') == 2
    fileID = fopen(full_file_path, 'rb');
    
    if fileID == -1
        error(['Failed to read ' full_file_path]);
    end

    raw = fread(fileID, inf, 'float32');
    
    % Close the file
    fclose(fileID);
else
    warning('The file %s does not exist.', full_file_path);
end

ground_truth = load(unit_info_file_path);
    
%% spike sorting with template_learning

% visualization of truth spike detection
sampling_rate = 24000;
spike_shape_samples = 32;
figure;
plot(raw, 'Color', [0 0 0 0.5]);
hold on;
title('Raw Signal with Spike Annotations');
xlabel('Sample Index');
ylabel('Signal Amplitude');
colormap_customized = lines(max(ground_truth.unit_index) + 1); 
for i = 1:length(ground_truth.sample_index)
    spikeTime = ground_truth.sample_index(i);
    unit = ground_truth.unit_index(i) + 1;
    timeRange = max(spikeTime-spike_shape_samples, 1):min(spikeTime+spike_shape_samples, length(raw));
    plot(timeRange, raw(timeRange), 'Color', colormap_customized(unit, :), 'LineWidth', 1.5);
end
hold off;
xlim([1000 30000]);

% calculate unit characteristics
units = unique(ground_truth.unit_index);
% for viz
window_length_firing_rate = 2; %1s
step_size_firing_rate = 1; %0.5s
for u = 1:length(units)
    figure('Position', [100, 100, 800, 600]);
    subplot(2, 3, 1);
    title('Average Spike Shape');
    xlabel('Time Point');
    ylabel('Amplitude');
    hold on;
    unit = units(u);
    unit_spikes = ground_truth.sample_index(ground_truth.unit_index == unit);
    spikes_shapes = zeros(spike_shape_samples + spike_shape_samples + 1, ...
        length(unit_spikes));
    
    for i = 1:length(unit_spikes)
        spikeTime = unit_spikes(i);
        if spikeTime > spike_shape_samples && (spikeTime + spike_shape_samples) <= length(raw)
            spikes_shapes(:, i) = raw((spikeTime-spike_shape_samples):(spikeTime+spike_shape_samples));
        end
    end
    
    plot(spikes_shapes, 'Color', [colormap_customized(unit + 1, :) 0.1]); % Assuming 'colormap_customized' is defined
    average_spike_shapes = mean(spikes_shapes, 2); % Average across columns (spikes)
    plot(average_spike_shapes, 'LineWidth', 2, 'Color', 'k'); % 'k' for black, or choose another color
    hold off; % Finish modifications to this subplot

    % do probability plot
    subplot(2, 3, 2);
    [D, AMPs, bins] = individualSpikePDF(spikes_shapes);
    hImg = imagesc(1:size(D, 2), AMPs, D);
    set(gca, 'YDir', 'normal'); 
    xlabel('Time Point');
    ylabel('Amplitude');
    title('Probability Density of Individual Spike Shapes');
    mask = D > ceil(size(spikes_shapes, 2) / bins / 10); 
    set(hImg, 'AlphaData', mask); 
    hold on
    plot(average_spike_shapes, '-', 'LineWidth', 2, 'Color', 'k');
    title('Probability Density of Spikes');
    hColorbar = colorbar;
    allSpikesFlat = spikes_shapes(:);
    ylabel(hColorbar, {'Amplitude Occurrence Density within 1 out of ' + string(bins) + ' bins', ...
           'between ' + string(min(allSpikesFlat)) + ' and ' + ...
           string(max(allSpikesFlat))}); 
    ylim([prctile(allSpikesFlat, 0.1) prctile(allSpikesFlat, 99.9)])
    hold off
    
    % ISI plot
    subplot(2, 3, 3);
    isi = diff(double(unit_spikes)) / sampling_rate; % in second
    proportionUnder3ms = sum(isi < 0.003) / length(isi);
            
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

    subplot(2, 3, 5);
    left = -0.05 * 1e3;
    right = 0.05 * 1e3;
    bin = 0.0005 * 1e3;
    spikeTimesIn_ms = double(unit_spikes) / sampling_rate * 1e3;
    diffST = spikeTimesIn_ms - spikeTimesIn_ms.';
    t = left:bin:right; 
    spikeTrain = histcounts(diffST, t); 
    spikeTrain(round((right - left) / bin / 2) + 1) = ...
        spikeTrain(round((right - left) / bin / 2) + 1) - length(spikeTimesIn_ms);
    plot(t(1:end-1) + 0.001 * 1e3 / 2, spikeTrain);

    xlabel('Lag (ms)');
    ylabel('Autocorrelation');
    title('Spike Time Autocorrelation');
    line([0 0], ylim, 'Color', 'r', 'LineStyle', '--'); % Zero lag line

    subplot(2, 3, 6);
    firing_rate = []; 
    for startSample = 1:round(step_size_firing_rate * sampling_rate):(length(raw) -...
            round(window_length_firing_rate * sampling_rate) + 1)
        endSample = startSample + round(window_length_firing_rate * sampling_rate) ;
        spikesCount = sum((startSample <= unit_spikes) ...
            & (unit_spikes <= endSample));
        firing_rate(end + 1) = spikesCount / window_length_firing_rate; 
    end
    timeVector = (1:length(firing_rate)) *...
        step_size_firing_rate; 
    windowSize = 10; 
    smoothedFiringRate = movmean(firing_rate,...
        windowSize);

    hold on;
    plot(timeVector, firing_rate, ...
        'b-', 'LineWidth', 1, 'DisplayName', 'Original Firing Rate');
    plot(timeVector, smoothedFiringRate, 'r-', 'LineWidth', 2, 'DisplayName', ...
        'Smoothed Firing Rate');
    hold off;
    xlabel('Time (s)');
    ylabel('Firing Rate (spikes/s)');
    title(['Firing Rate (' num2str(window_length_firing_rate) 's window)']);
    legend('show');    

    filename = fullfile(save_dir, ['unit' num2str(u) '_characteristic_' session_name ...
        dateTimeNow '.svg']);
        saveas(gcf, filename, 'svg');         
end

% visualize the data and unit identification
[spike_output] = template_learning_and_matching(raw, sampling_rate, ...
    save_dir, 5e3:6e3, 0.2, 99);

%% compare performance metrics
spike_output = load(['spikeForest_output/' ...
    'ch_1_sorting_results_2024-03-28_13-34-48.mat']);
spike_output = spike_output.spike_2_export;
delta_t = 0.003; % if an event is detected within 5 ms, we consider a match
delta_sample = round(sampling_rate * delta_t);
% mapping(1) = 2 means that 2nd sorted unit correspond to 1st ground truth
% unit
mapping = zeros(1, length(units));
% precision is 1 - false positive
% recall is 1 - false negative 
% accuracy balance precision and recall
precision = zeros(1, length(units));
recall = zeros(1, length(units));
accuracy = zeros(1, length(units));
for u = 1:length(units)
    unit = units(u);
    unit_spikes = ground_truth.sample_index(ground_truth.unit_index == unit);
    spikes_shapes = zeros(spike_shape_samples + spike_shape_samples + 1, ...
        length(unit_spikes));
    for i_spike = 1:length(unit_spikes)
        spikeTime = unit_spikes(i_spike);
        if spikeTime > spike_shape_samples && (spikeTime + spike_shape_samples) <= length(raw)
            spikes_shapes(:, i_spike) = raw((spikeTime-spike_shape_samples):(spikeTime+spike_shape_samples));
        end
    end
    max_accuracy = -Inf;
    for i_sorted = 1:length(spike_output)
        nMatch = 0;
        sorted_spikes = [spike_output(i_sorted).spike_time];
        % match_to_ground_truth = []; 
        for i_ground_truth = 1:length(unit_spikes)
            ground_truth_spike_time = unit_spikes(i_ground_truth);
            % Find if there's a sorted spike within ±delta_t of the ground truth spike
            if any(abs(int64(sorted_spikes) - ground_truth_spike_time) < delta_sample)
                nMatch = nMatch + 1;
                % match_to_ground_truth = [match_to_ground_truth 
                %     sorted_spikes(abs(int64(sorted_spikes) - ground_truth_spike_time) < delta_sample)];
            end
        end
        % nMatch = length(unique(match_to_ground_truth));
        nMiss = length(unit_spikes) - nMatch; % Number of missed events / false netative
        nFP = length(sorted_spikes) - nMatch; % Number of false positive events
        current_precision = nMatch / (nMatch + nFP);
        current_recall = nMatch / length(unit_spikes);
        current_accuracy = nMatch / (length(unit_spikes) + length(sorted_spikes) - nMatch);

        if current_accuracy > max_accuracy
            precision(u) = current_precision;
            recall(u) = current_recall;
            accuracy(u) = current_accuracy;
            mapping(u) = i_sorted;
            max_accuracy = current_accuracy;
        end
    end
end

% compare ground truth spike and detected spike on top of the raw signals
% visualization of truth spike detection
sampling_rate = 24000;
spike_shape_samples = 32;
figure;
subplot(2,1,1)
plot(raw, 'Color', [0 0 0 0.5]);
hold on;
title('Raw Signal with Spike Annotations (truth)');
xlabel('Sample Index');
ylabel('Signal Amplitude');
colormap_customized = lines(max(ground_truth.unit_index) + 1); 
colormap_customized = [colormap_customized, [0.5,0.5,0.5]'];

for i = 1:length(ground_truth.sample_index)
    spikeTime = ground_truth.sample_index(i);
    unit = ground_truth.unit_index(i) + 1;
    timeRange = max(spikeTime-spike_shape_samples, 1):min(spikeTime+spike_shape_samples, length(raw));
    plot(timeRange, raw(timeRange), 'Color', colormap_customized(unit, :), 'LineWidth', 1.5);
end
hold off;
xlim([1000 30000]);
subplot(2,1,2);
plot(raw, 'Color', [0 0 0 0.5]);
hold on;
title('Raw Signal with Spike Annotations (detected)');
xlabel('Sample Index');
ylabel('Signal Amplitude');

for i_sorted = 1:length(mapping)
    detected_spikes = [spike_output(mapping(i_sorted)).spike_time];
     for i_spike = 1:length(detected_spikes)
        spikeTime = detected_spikes(i_spike);
        if spikeTime > 1 && (spikeTime + spike_shape_samples * 2) <= length(raw)
            timeRange = max(spikeTime, 1):min(spikeTime+spike_shape_samples+spike_shape_samples, length(raw));
            plot(timeRange, raw(timeRange), 'Color', colormap_customized(i_sorted, :), ...
                'LineWidth', 1.5); % +1 to avoid index 0
        end
    end
end
hold off;
xlim([1000 30000]);


