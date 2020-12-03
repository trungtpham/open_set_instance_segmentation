function scores = compute_pixels_to_boxcenters_fitting_scores(det_bboxes, im_size)

num_dets = size(det_bboxes,1);
H = im_size(1);
W = im_size(2);
[py, px] = find(ones(H,W));

% Compute detection box centers and radii
det_cx = det_bboxes(:,1) + 0.5.*det_bboxes(:,3);
det_cy = det_bboxes(:,2) + 0.5.*det_bboxes(:,4);
det_rx = (0.5*det_bboxes(:,3)).^1.0;
det_ry = (0.5*det_bboxes(:,4)).^1.0;

% Compute normalize distances from pixel locations to box centers
xdist = pdist2(det_cx, px, 'squaredeuclidean')./repmat(2.*(det_rx.^2), 1, W*H);
ydist = pdist2(det_cy, py, 'squaredeuclidean')./repmat(2.*(det_ry.^2), 1, W*H);
spdist = xdist + ydist;
scores = -reshape(spdist, [num_dets, W*H]);

end
