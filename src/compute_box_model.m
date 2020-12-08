function [iou_mat, box_model_scores] = compute_box_model(detections, features)
if isempty(detections.bboxes)
    box_model_scores = [];
    iou_mat = [];
    return;
end

% Compute box centerness scores
[box_center_fitness_scores, region_center_fitness_scores] = compute_box_spatial_model(detections, features);

% Compute iou fitness scores between boxes and regions
% Compute overlap between boxes and regions
region_bboxes = cat(1,features.bboxes);
region_bboxes = max(region_bboxes,1);
iou_mat = bboxOverlapRatio(detections.bboxes, region_bboxes);
% Compute area ratios (given two regions which have similar iou with a
% bounding box, the smaller region is preferred.
load('mask_area_prior.mat');
bbox_areas = detections.bboxes(:, 3).* detections.bboxes(:,4);
class_ids = detections.class_ids;
ratios = min(1.0, 1.0.*mask_area_prior(class_ids));
expected_mask_areas = bbox_areas.*ratios;
region_areas = features.areas';
area_ratios = region_areas ./ expected_mask_areas;
mean_ratio = 1.0;
iou_scores = iou_mat.* exp(-(area_ratios - mean_ratio).^2/0.5);
iou_fitness_scores = log(iou_scores).*region_areas;

num_nodes = length(features.areas);
det_confidences = repmat(detections.scores, 1, num_nodes);

box_model_scores = log(det_confidences).*region_areas... 
                 + iou_fitness_scores...   
                 + 0.5.*box_center_fitness_scores...
                 + 0.75.*region_center_fitness_scores;
box_model_scores = box_model_scores';
iou_mat = iou_mat';

end

function [box_center_fitness_scores, region_center_fitness_scores] = compute_box_spatial_model(detections, features)
global pixel2boxcenter_fitting_scores
num_nodes = length(features.areas);
num_dets = size(detections.bboxes, 1);
box_center_fitness_scores = zeros(num_dets, num_nodes);
for i=1:num_nodes
    box_center_fitness_scores(:, i) = sum(pixel2boxcenter_fitting_scores(:,features.masks{i}), 2);
end

% Get bboxes from the segmentation tree
region_bboxes = cat(1,features.bboxes);
region_bboxes = max(region_bboxes,1);
region_cx = region_bboxes(:,1) + 0.5*region_bboxes(:,3);
region_cy = region_bboxes(:,2) + 0.5*region_bboxes(:,4);

% For each region, first compute distance center to detection bounding box
% coordinates.
region_areas = features.areas';
det_bboxes = detections.bboxes;
dx_tl = pdist2(region_cx, det_bboxes(:, 1), 'squaredeuclidean');
dy_tl = pdist2(region_cy, det_bboxes(:, 2), 'squaredeuclidean');
dx_br = pdist2(region_cx, det_bboxes(:, 1) + det_bboxes(:, 3), 'squaredeuclidean');
dy_br = pdist2(region_cy, det_bboxes(:, 2) + det_bboxes(:, 4), 'squaredeuclidean');
dd    = 0.5*(sqrt(dx_tl + dy_tl) + sqrt(dx_br + dy_br));
size_sigma = 0.125*sum(features.im_size);
region_center_fitness_scores = -(dd'./size_sigma).*region_areas;

end
