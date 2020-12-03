clear all

%% Load coco related utils
load coco_class_names.mat
load coco_class_id_map
load map_imageId_filename
load coco_val_img_ids

%% Path to coco data
data_path = 'datasets/ms_coco_samples/';
images_files = dir(strcat(data_path, 'images'));
images_files(1:2) = [];

for img_idx=1:length(images_files)
    fprintf('segmenting image... %d \n', img_idx);
    %% Get filename
    img_filename = images_files(img_idx).name;
    img_name = strtok(img_filename, '.');

    %% Load pre-computed hierarchical tree
    load(strcat(data_path, 'features/', img_name, '.mat'));
    load(strcat(data_path, 'trees/', img_name, '.mat'));

    %% Load input image
    rgb_im = imread(strcat(data_path, 'images/', img_name, '.jpg'));
    [H, W, C] = size(rgb_im);

    %% Load pre-computed mask-rcnn detections and masks
    load(strcat(data_path, 'detections/', img_name,'.mat'));
    % Remove detections whose scores are lower than a threshold
    bad_det_ids = detections.scores > 0.25;
    detections.bboxes =  detections.bboxes(bad_det_ids,:);
    detections.masks =  detections.masks(bad_det_ids);
    detections.scores =  detections.scores(bad_det_ids);
    detections.class_ids =  detections.class_ids(bad_det_ids);
    detections.areas =  detections.areas(bad_det_ids);
    detections.image_ids =  detections.image_ids(bad_det_ids);
    
    %% Optimise instance segmentation using monte carlo sampling method.  
    optimiser = 'mc';
    tic
    [segmentation, labels] = instance_segmentation(tree, features, detections, optimiser);
    toc
    
    %% Visualize segmentation results.
    segmentation = double(segmentation);
    seg_ids = unique(segmentation);
    L = labels(seg_ids);
    
    % Relabel starting from 1, 2 
    for i=1:length(seg_ids)
        segmentation(segmentation == seg_ids(i)) = i;
    end
    num_colors = length(seg_ids);
    colors = distinguishable_colors(num_colors);
    seg_color = labeloverlay(rgb_im, segmentation, 'Colormap', colors, 'Transparency', 0.3);

    % Visualize detection ontop of segmentation
    num_detections = size(detections.bboxes, 1);
    for i=1:num_detections
        bb = detections.bboxes(i,:);
        centroid = [bb(1)+bb(3)/2, bb(2) + bb(4)/2];
        cid = coco_class_id_map(detections.class_ids(i));
        classname = class_names{cid};
        score = detections.scores(i);
        if ismember(i, L)
            caption = strcat(num2str(i), '-', classname, '-', num2str(score));
            seg_color = insertShape(seg_color, 'Rectangle', bb, 'Color', 'blue', 'LineWidth', 2);
            seg_color = insertText(seg_color, [bb(1), bb(2)], caption, 'FontSize', 10, 'TextColor', 'white', 'BoxColor', 'green', 'BoxOpacity',0.0);
        else
            caption = strcat(num2str(i), '-FP-', classname, '-', num2str(score));
            seg_color = insertShape(seg_color, 'Rectangle', bb, 'Color', 'red', 'LineWidth', 1);
            seg_color = insertText(seg_color, [bb(1), bb(2)], caption, 'FontSize', 10, 'TextColor', 'white', 'BoxColor', 'green', 'BoxOpacity',0.0);
        end
    end
    figure(1);imshow(seg_color);
    drawnow;
    pause;
end

