%% description
% This is a script for loading data, using the template learning and
% matching detection method to detect spikes channelwise, visualizing
% detected spikes, and conducting global merge for microwires on the same
% shank.

addpath('bci2000_fc');

% add data path, the LFP and SUA (single unit activity) are recorded from
% BJH035, during the first session of a memory task (BLAES_ImmediateTest)
% ECOGS001R01_1 records SUA, ECOGS001R01 records LFP from macro electrode
% along with eye tracking. The two recordings can be aligned with the sync
% pulses. Note that ch 21 is the sync pulse in SUA recording, DC04 is the
% sync pulse in LFP recording.

% parameters
n_ch_per_micro_contact = 8;

% load necessary data
[signal_mic, states_mic, parameters_mic] = load_bcidat('bci2000_data/ECOGS001R01_1.dat');
contact_data_table = load('bci2000_data/contact_data_table.mat');
contact_data_table = contact_data_table.contact_data_table;
shank_id_micro_contact = contact_data_table.ShankID(contact_data_table.ContactID == 99);
micro_ch_names = cell(round(length(shank_id_micro_contact) * n_ch_per_micro_contact), 1);

for i_shank_id_micro_contact = 1:length(shank_id_micro_contact)
    for i = 1:n_ch_per_micro_contact
        micro_ch_names{(i_shank_id_micro_contact - 1) * ...
        n_ch_per_micro_contact + i} = [shank_id_micro_contact{i_shank_id_micro_contact}, num2str(i)];
    end
end

% we will sort one shank
i_shank_id_micro_contact = 1;
channel_start = (i_shank_id_micro_contact - 1) * n_ch_per_micro_contact + 1;
channel_end = (i_shank_id_micro_contact - 1) * n_ch_per_micro_contact + 8;
signal_mic_subset = double(signal_mic(:, channel_start:channel_end));
save_dir = 'bci2000_output/';

% for testing
% signal_mic_subset = signal_mic_subset(:, 3:7);

% start sorting
[spike_output] = template_learning_and_matching(signal_mic_subset, parameters_mic.SamplingRate.NumericValue,...
    save_dir, 5e3:6e3, 0.2, 99);


