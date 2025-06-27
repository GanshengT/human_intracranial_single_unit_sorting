function [D, AMPs, bins] = individualSpikePDF(individualSpikeShapes, winP, doploting, bins)
    if nargin < 2 || isempty(winP)
        winP = 4; % Default window percentage
    end
    if nargin < 3 || isempty(doploting)
        doploting = 0; % Default plotting option
    end
    if nargin < 4 || isempty(bins)
        bins = 200; % Default number of bins
    end

    A = individualSpikeShapes;
    added = min(A(:));
    A = A - added; % Shift amplitude data to start from 0
    res = (max(A(:)) - min(A(:))) / bins;
    A = round(A / res); % Convert amplitudes to bin indices
    
    D = zeros(bins, size(A, 1));
    
    for i_time = 1:size(A, 1)
        for iBin = 1:bins
            D(iBin, i_time) = sum(A(i_time,:) == iBin);
        end
    end
    
    winSize = round(bins * winP / 100);
    gaussFilter = gausswin(winSize) / sum(gausswin(winSize)); 
    for i_time = 1:size(A, 1)
        D(:, i_time) = conv(D(:, i_time), gaussFilter, 'same');
    end

    AMPs = ((1:bins) - 1) * res + added;
    
    % Plotting
    if doploting
        figure;
        hImg = imagesc(1:size(A, 1), AMPs, D);
        set(gca, 'YDir', 'normal'); % Correct the y-axis direction
        xlabel('Time Point');
        ylabel('Amplitude');
        title('Probability Density of Individual Spike Shapes');
        
        mask = D > 1; 
        set(hImg, 'AlphaData', mask); % Apply the mask as transparency data
        
        colorbar;
    end
end
