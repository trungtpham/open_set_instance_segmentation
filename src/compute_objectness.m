% ------------------------------------------------------------------------ 
% This function computes, for each image region, an "objectness" score that
% describes how likely a region to be an object (thing or stuff). Here the
% objectness score is computed based on image boundary scores.
% ------------------------------------------------------------------------ 

function [objectness_scores, boundary_scores] = compute_objectness(features)
num_regions = length(features.areas);
boundary_scores = compute_boundary_model(features);
areas = features.areas;
im_size = features.im_size;
% Penalize small objects
obj_size_prior = zeros(num_regions,1);
object_size_ths = 0.02; % 2% of the image size.
small_object_ids = areas/prod(im_size) < object_size_ths;
obj_size_prior(small_object_ids) = -((areas(small_object_ids)/prod(im_size) - object_size_ths).^2 / 1.0);
constant_penalty = log(0.05);
objectness_scores = (obj_size_prior + constant_penalty).*areas + boundary_scores;
end
