% Generate UCMs, trees, and features
dataset_path = 'datasets/ms_coco_samples/'; 
img_dir = strcat(dataset_path, 'images/');
images_files = dir(img_dir);
images_files(1:2) = [];
for i=1:length(images_files)
    img_name = images_files(i).name;
    img_name = strtok(img_name, '.');
    
    ucm_file_name_dest = strcat(dataset_path, 'ucms/', img_name, '.mat');
    load(ucm_file_name_dest);
    
    [tree, base, hseg] = ucm2tree(ucm2);
    ths = 0.05:0.025:1.0;
    N = size(tree,2);
    start_ths = zeros(N,1);
    end_ths = zeros(N,1);
    for j=1:size(hseg,3)
        L = unique(hseg(:,:,j));
        end_ths(L) = ths(j);
    end
    for j=1:N
        C = tree(j).children;
        if isempty(C)
            start_ths(j) = 0;
        elseif length(C) == 1
            start_ths(j) = start_ths(C);
        else
            start_ths(j) = mean(end_ths(C));
        end
    end
    features = struct;
    features.start_ths = start_ths;
    features.end_ths = end_ths;
    features.areas = [];
    features.bboxes = [];
    features.masks = cell(0,1);
    features.im_size = size(base);
    region_idx = 0;
    for j=1:size(hseg,3)
        seg = hseg(:,:,j);
        seg = seg - region_idx;
        feat = regionprops(seg, 'PixelIdxList', 'Area', 'BoundingBox');
        areas = [feat.Area];
        bboxes = [feat.BoundingBox];
        bboxes = reshape(bboxes, [4, length(feat)]);
        masks = {feat.PixelIdxList};
        masks = masks';
        features.areas = [features.areas; areas'];
        features.bboxes = [features.bboxes; bboxes'];
        features.masks =  [features.masks; masks];
        region_idx = max(max(hseg(:,:,j)));
    end
    
    %% save segmentation tree
    tree_output_path = strcat(dataset_path, 'trees/');
    if ~exist(tree_output_path, 'dir')
       mkdir(tree_output_path)
    end
    tree_out_file = strcat(tree_output_path, img_name, '.mat');
    save(tree_out_file, 'tree');
    
    %% save features
    features_output_path = strcat(dataset_path, 'features/');
    if ~exist(features_output_path, 'dir')
       mkdir(features_output_path)
    end
    feature_out_file = strcat(features_output_path, img_name, '.mat');
    save(feature_out_file, 'features');
    fprintf('Done generating tree and features for image %d \n', i);
end

function [tree, base, hseg] = ucm2tree(ucm)
% Convert ucm to tree
ths = 0.05:0.025:1.0;
max_cur_label = 0;
hseg = zeros([floor(size(ucm)/2), length(ths)]);
for i=1:length(ths)
    seg = ucm2segmentation(ucm, ths(i));
    seg = seg + max_cur_label;
    hseg(:,:,i) = seg;
    max_cur_label = max(seg(:));
end

tree = struct;
for i=2:length(ths)
    base = hseg(:,:,i-1);
    labels = unique(hseg(:,:,i));
    for s=1:length(labels)
        parent = labels(s);
        idx = hseg(:,:,i) == parent;
        children = unique(base(idx));
        tree(parent).parent = parent;
        tree(parent).children = children; 
    end
end
if length(unique(hseg(:,:,end))) > 1
    root = max(hseg(:)) + 1;
    tree(root).parent = root;
    tree(root).children = unique(hseg(:,:,end));
end
base = hseg(:,:,1);
end

function seg = ucm2segmentation(ucm, th)
% Get segmentation from ucm
tmp_ucm = ucm;
tmp_ucm(tmp_ucm<th) = 0;
tmp_ucm(1:2:end,1:2:end)=1; % Make the gridbmap connected
labels = bwlabel(tmp_ucm' == 0, 8); % Transposed for the scanning to be from
                                    % left to right and from up to down
labels = labels';
seg = labels(2:2:end, 2:2:end);
end