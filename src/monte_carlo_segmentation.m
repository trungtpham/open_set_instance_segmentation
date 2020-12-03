%--------------------------------------------------------------------------
% Energy minimization based segmentation using Monte Carlo sampling method.
%--------------------------------------------------------------------------

function [segmentation, node_binary_labels, node_instance_labels, iou_matrix, features] = monte_carlo_segmentation(...
    tree, features, segmentation_energies, detections, iou_mat, init_seg, params)
                                                                                
% Control random seed
rng(1);

%% Compute nodes' parents
child_father_tab = zeros(length(tree), 1);
for i=1:length(tree)
    children = tree(i).children;
    if ~isempty(children)
        child_father_tab(children) = i;
    end
end

%% Facts
num_detections = size(detections.bboxes,1);
unknown_label = num_detections + 1;
num_nodes = length(segmentation_energies);
empty_mask = (zeros(size(init_seg), 'uint32'));
max_node_ID = num_nodes;

%% Compute init energy and node labels
max_num_nodes = num_nodes + 100;
energies = zeros(max_num_nodes,1);
node_instance_labels = zeros(max_num_nodes,1);
node_binary_labels = zeros(max_num_nodes, 1);
iou_matrix = zeros(max_num_nodes, num_detections);
iou_matrix(1:num_nodes, 1:num_detections) = iou_mat;
[temp_energies, labels] = min(segmentation_energies, [], 2);
energies(1:num_nodes) = temp_energies;
node_instance_labels(1:num_nodes) = labels;
init_regions = unique(init_seg);
init_regions(init_regions==0) = [];
node_binary_labels(init_regions) = 1;
current_energy = sum(energies(node_binary_labels==1));
current_seg = uint32(init_seg);
adj_graph = imRAG(current_seg, 1);

%% Simulated Annealing parameters
max_iterations = params.max_iterations;
temperature = params.temperature;
max_rejections = params.max_rejections;
decay = params.decay;
T = temperature;

%% Optimisation loop
num_rejects = 0;
for s=1:max_iterations
    if num_rejects > max_rejections
        break;
    end
    
    % Decrease temperator
    T = T*decay;
    
    % Randomly sample a move
    rand_toss = randsample([1 2], 1, 'true', [0.5 0.5]);
    switch rand_toss
        case 1 
            move = 'split';
        case 2
            move = 'merge';
    end

    % Current node labels
    current_node_labels = node_binary_labels.*node_instance_labels;
   
    %% Merge process
    if (strcmp(move, 'merge'))
        % Sample a detection k
        k = datasample(1:num_detections, 1);
        % Get nodes assigned to k
        node_candidates = find(current_node_labels==k, 1);        
        % Get iou scores between detection k and regions
        iou_scores = iou_matrix(:,k).*(current_node_labels>0);
        % Sample the first region
        if isempty(node_candidates)
            [~, candidate] = max(iou_scores);
            % If the candidate has assigned to a detection different from
            % k, select another candidate.
            if (current_node_labels(candidate) ~= unknown_label)
                iou_scores(candidate) = 0;
                [~, candidate] = max(iou_scores);
            end
        else
            [~, idx] = max(iou_scores(node_candidates));
            candidate = node_candidates(idx);
        end
        % Get neighbor regions using adjaceny graph
        src = (adj_graph(:,1) == candidate);
        dst = (adj_graph(:,2) == candidate);
        candidate_neighbors = [adj_graph(src,2); adj_graph(dst,1)];
   
        % Search for the best candidate for merging
        if ~isempty(candidate_neighbors)
            merge_candidates = [datasample(candidate_neighbors, 1) candidate];
        else
            num_rejects = num_rejects + 1;
            continue;
        end
        
        % Compute features for the newly merged region.
        merged_pixels = cell2mat(features.masks(merge_candidates));
        merged_mask = empty_mask;
        merged_mask(merged_pixels) = 1;
        bbox = regionprops(merged_mask, 'BoundingBox').BoundingBox;
        temp_features.areas = length(merged_pixels);
        temp_features.bboxes = round(bbox);
        temp_features.masks = num2cell(merged_pixels, 1);
        temp_features.end_ths = features.end_ths(candidate);
        temp_features.start_ths = features.start_ths(candidate);
        temp_features.im_size = features.im_size;
        
        % Compute objectness score for the new region
        [objectness_scores, ~]  = compute_objectness(temp_features);
        
        % Compute box fitting score
        [temp_iou_mat, box_model_scores] = compute_box_model(detections, temp_features);
        
        % Build energy matrix for optimisation
        temp_energies = [-box_model_scores -objectness_scores];
        
        % Compute energy difference between the old and new segmentations
        [temp_energies, temp_node_instance_label] = min(temp_energies);
        % If the new region is assigned to a detection different from k,
        % reject this merge process.
        if temp_node_instance_label ~= k
            num_rejects = num_rejects + 1;
            continue
        end
        energy_diff = sum(energies(merge_candidates)) - temp_energies;
        % Compute acceptance probability
        ap = exp(energy_diff/T);
        if ap>rand
            % Increase the number of nodes
            max_node_ID = max_node_ID + 1;
            % Update segmentation
            current_seg(merged_pixels) = max_node_ID;
            % Update adjacency graph
            adj_graph = imRAG(current_seg, 1);
            % Update current energy
            current_energy = current_energy - energy_diff;
            energies(max_node_ID) = temp_energies;
            % Update node labels
            node_binary_labels(merge_candidates) = 0;
            node_binary_labels(max_node_ID) = 1;
            node_instance_labels(max_node_ID) = temp_node_instance_label;
            % Update iou matrix
            iou_matrix(max_node_ID, :) = temp_iou_mat;
            
            % Update features struct
            features.areas(max_node_ID) = temp_features.areas;
            features.masks(max_node_ID) = temp_features.masks;
            features.bboxes(max_node_ID,:) = temp_features.bboxes;
            features.end_ths(max_node_ID) = temp_features.end_ths;
            features.start_ths(max_node_ID) = temp_features.start_ths;
            % Reset reject counts
            num_rejects = 0;
       else
            num_rejects = num_rejects + 1;
       end    
    end
    
    %% Split process   
    if (strcmp(move, 'split'))    
        % Sample a detection k
        k = datasample(1:num_detections, 1);       
        % if the detection k is being used in the current segmentation,
        % terminate the split process, otherwise select a region in the
        % segmentation tree that has higher overlap with the detection k.
        if isempty(find(current_node_labels==k, 1))
            [~, picked_node] = max(iou_matrix(:,k));
            if current_node_labels(picked_node) > 0
                num_rejects = num_rejects + 1;
                continue;
            end
        else
            num_rejects = num_rejects + 1;
            continue;
        end
        % Compute split regions
        overlap_nodes = unique(current_seg(features.masks{picked_node}));
        overlap_pixels = cell2mat(features.masks(overlap_nodes));
        split_pixels = setdiff(overlap_pixels, features.masks{picked_node});
        if (~isempty(split_pixels))
            split_mask = empty_mask;
            split_mask(split_pixels) = 1;
            split_mask = split_mask.*current_seg;
            stats = regionprops(split_mask, 'Area', 'BoundingBox', 'PixelIdxList');
            idx = [stats.Area] > 0; 
            stats = stats(idx);
           
            areas = [stats.Area];
            bboxes = [stats.BoundingBox];
            bboxes = reshape(bboxes, [4, length(stats)]);
            masks = {stats.PixelIdxList};
            masks = masks';
            temp_features.areas = [areas'];
            temp_features.bboxes = [bboxes'];
            temp_features.masks =  [masks];
            temp_features.end_ths = repmat(mean(features.end_ths(overlap_nodes)), [length(areas), 1]);
            temp_features.start_ths = repmat(mean(features.start_ths(overlap_nodes)), [length(areas), 1]);
            temp_features.im_size = features.im_size;
                
            % Compute objectness score for the new region
            [objectness_scores, ~]  = compute_objectness(temp_features);
            
            % Compute box fitting score
            [temp_iou_mat, box_model_scores] = compute_box_model(detections, temp_features);
            
            temp_energies = [-box_model_scores -objectness_scores];
            [temp_energies, temp_node_instance_label] = min(temp_energies, [], 2);
            energy_diff = sum(energies(overlap_nodes)) - (energies(picked_node) + sum(temp_energies));

            ap = exp(energy_diff/T);
            if ap > rand        
                % Update energy
                num_new_nodes = length(areas);
                current_energy = current_energy - energy_diff;
                energies(max_node_ID+1:max_node_ID+num_new_nodes) = temp_energies;
                % Update node labels
                node_binary_labels(picked_node) = 1;
                node_binary_labels(overlap_nodes) = 0;
                node_binary_labels(max_node_ID+1:max_node_ID+num_new_nodes) = 1;
                node_instance_labels(max_node_ID+1:max_node_ID+num_new_nodes) = temp_node_instance_label;
                % Update iou matrix
                iou_matrix(max_node_ID+1:max_node_ID+num_new_nodes, :) = temp_iou_mat;
                % Update node features
                features.areas(max_node_ID+1:max_node_ID+num_new_nodes) = temp_features.areas;
                features.masks(max_node_ID+1:max_node_ID+num_new_nodes) = temp_features.masks;
                features.bboxes(max_node_ID+1:max_node_ID+num_new_nodes,:) = temp_features.bboxes;
                features.end_ths(max_node_ID+1:max_node_ID+num_new_nodes) = temp_features.end_ths;
                features.start_ths(max_node_ID+1:max_node_ID+num_new_nodes) = temp_features.start_ths;
                
                % Update current segmentation
                current_seg(features.masks{picked_node}) = picked_node;
                max_node_ID = max_node_ID + 1;
                for i=1:num_new_nodes
                    current_seg(temp_features.masks{i}) = max_node_ID;
                    max_node_ID = max_node_ID + 1;
                end
                % Update adjacency graph
                adj_graph = imRAG(current_seg, 1);
                % Reset reject counts
                num_rejects = 0;
            else
                num_rejects = num_rejects + 1;
            end    
        end
    end
end
segmentation = current_seg;
end



