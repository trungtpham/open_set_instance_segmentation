function [segmentation, node_instance_labels] = instance_segmentation(tree, features, detections, optimiser)
% Input: 
%   tree: segmentation hierarchical tree
%   features: contain features for each node (region) such as area,
%   boundary scores.
%   detections: detection bounding boxes returned a detection network such as
%   ssd, yollo or faster rcnn
%   optimiser: method to optimise the segmentation. tree cut (aka tc) (fast)
%   or monter carlo (aka mc) (slow but more accurate). By default a tree
%   cut method is used.
% Ouput:
%   segmentation: output segmentation 
%   node_labels: 

num_objects = size(detections.bboxes, 1);
unknown_class_id = num_objects + 1;

%% Compute objectness scores
[objectness_scores, ~]  = compute_objectness(features);

%% Compute bounding model
if num_objects > 0
    global pixel2boxcenter_fitting_scores;
    pixel2boxcenter_fitting_scores = compute_pixels_to_boxcenters_fitting_scores(detections.bboxes, features.im_size);
    [box_iou_mat, box_model_scores] = compute_box_model(detections, features);
    iou_mat = box_iou_mat;
    segmentation_scores = [box_model_scores objectness_scores];
else
    segmentation_scores = objectness_scores;
end

%% Run tree cut using dynamic programming
[cut, node_instance_labels] = tree_cut_dynamic_programming(tree, segmentation_scores);
selected_nodes = find(cut==1);
im_size = features.im_size;
segmentation = zeros(im_size);
for i=1:length(selected_nodes)
    idx = features.masks{selected_nodes(i)};
    segmentation(idx) = selected_nodes(i);   
end

%% Run Monte Carlo simulation
if num_objects > 0 && strcmp(optimiser, 'mc')
    temperature = 10;
    decay = 0.99;
    max_iterations = max(1000,num_objects*10);
    params.max_iterations = max_iterations;
    params.max_rejections = 100;
    params.temperature = temperature;
    params.decay = decay;
    
    init_seg = segmentation;
    segmentation_energies = -segmentation_scores;
    [segmentation, node_binary_labels, node_instance_labels, iou_mat, ~] =...
        monte_carlo_segmentation(tree, features, segmentation_energies,... 
                                 detections, iou_mat, init_seg, params);                                                                         
    
    % If there are more than 1 regions assigned to a single detection
    % heuristically select the region with highest overlap. Other regions
    % will be assigned to unknown class.
     L = node_binary_labels.*node_instance_labels;
     
     if ~isequal(find(L>0), unique(segmentation))
         msg = 'Error occurred.';
         error(msg)
     end
     for i=1:num_objects
         regions = find(L==i);
         if length(regions) > 1
            ious = iou_mat(regions, i);
            [~, idx] = max(ious);
            node_instance_labels(setdiff(regions, regions(idx))) = unknown_class_id;
         end
     end                                                
end

end