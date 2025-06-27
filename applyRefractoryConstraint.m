function xOpt = applyRefractoryConstraint(solution, expectation_decrease_objective_fn, refractory_period)
    [numTimePoints, numTemplates] = size(solution);
    modifiedSolution = zeros(size(solution));

    for iTemplate = 1:numTemplates
        indices = find(solution(:, iTemplate) == 1); 
        if isempty(indices)
            continue; 
        end

        clusters = {indices(1)};
        for i = 2:length(indices)
            if indices(i) - indices(i-1) <= refractory_period
                clusters{end}(end+1) = indices(i);
            else
                clusters{end + 1} = indices(i); 
            end
        end
        
        % Process each cluster
        for iCluster = 1:length(clusters)
            clusterIndices = clusters{iCluster};
            if length(clusterIndices) == 1
                modifiedSolution(clusterIndices, iTemplate) = 1; % Keep if solo in cluster
                continue;
            end
            
            
            [~, maxIdx] = max(expectation_decrease_objective_fn(clusterIndices, iTemplate));
            bestIndex = clusterIndices(maxIdx);
            modifiedSolution(bestIndex, iTemplate) = 1; % Keep only this 1
        end
    end

    xOpt = reshape(modifiedSolution, numTimePoints*numTemplates, 1);
end