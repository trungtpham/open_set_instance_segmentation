function [boundary_scores]  = compute_boundary_model(features)
start_ths = features.start_ths;
end_ths = features.end_ths;
areas = features.areas;
sigma_i = 0.3;
sigma_o = 0.7;
scores = exp(-(start_ths - 0).^2./sigma_i^2).*exp(-(end_ths - 1).^2./sigma_o^2);
boundary_scores = log(scores).*areas;

end