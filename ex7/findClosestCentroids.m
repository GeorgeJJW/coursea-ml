function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% obtain training example size
m = size(X, 1);

% create distance matrices
D1 = zeros(K, size(X, 2));
D2 = zeros(K, 1);

% for each example
for i = 1:m
  
  % for each centroid
  for j = 1:K
    % compute distance to each centroids
    D1(j, :) = X(i, :) - centroids(j, :);
  endfor
  
  % square each distance components
  D1 = D1 .^ 2;
  
  % determine the smallest squared distance
  D2 = sum(D1, 2);
  [d_val, d_idx] = min(D2);
  idx(i) = d_idx;  
  
endfor

% =============================================================

end

