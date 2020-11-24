function [ d ] = my_mahalanobis( feature_vector, mean, covariance)
% feature_vector and mean 2x1 and Covariance 2x2
   d = ( (feature_vector - mean)'*inv(covariance)*(feature_vector - mean));
end 