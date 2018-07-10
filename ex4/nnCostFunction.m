function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% add bias units to the input feature layer
A1 = [ones(m, 1) X];

% activate second layer
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);

% add bias units to the second layer
A2 = [ones(m, 1) A2];

% activate output layer
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

% calculate cost
y_eye = eye(num_labels);

for i = 1:m
  
  % convert y label to the appropriate identity vector
  y_cur = y_eye(:, y(i, 1));
  
  % loop through each output unit
  for c = 1:num_labels
    
    % calculate cost for that example
    cost = (-y_cur(c, 1) * log(A3(i, c))) - ((1 - y_cur(c, 1)) * log(1 - A3(i, c)));
    
    % accumulate cost
    J += cost;
  
  endfor

endfor

% weight cost by example size
J = J / m;

% compute regularization term for cost function
t1_size = size(Theta1);
t2_size = size(Theta2);
ss_t1 = 0;
ss_t2 = 0;
 
for j1 = 1:t1_size(1, 1)
  for k1 = 2:t1_size(1, 2)
    r1 = Theta1(j1, k1) ^ 2;
    ss_t1 += r1;
  endfor
endfor

for j2 = 1:t2_size(1, 1)
  for k2 = 2:t2_size(1, 2)
    r2 = Theta2(j2, k2) ^ 2;
    ss_t2 += r2;
  endfor
endfor

r = (lambda / (2 * m)) * (ss_t1 + ss_t2);

% obtain cost with regularization
J = J + r;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 ansd Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t = 1:m
  
  % set input layer values
  a_1 = X(t, :);
  a_1 = a_1(:);
  
  % add bias unit
  a_1 = [1 ; a_1];  
  
  % compute activations
  z_2 = a_1' * Theta1';
  a_2 = sigmoid(z_2);
  a_2 = a_2(:);
  a_2 = [1 ; a_2];
  z_3 = a_2' * Theta2';
  a_3 = sigmoid(z_3);
  a_3 = a_3(:);
  
  % convert y label to the appropriate identity vector
  y_t = y_eye(:, y(t, 1));
  
  % compute error term for output layer
  d_3 = a_3 - y_t;
  
  % compute error term for the hidden layer
  d_2 = (Theta2' * d_3) .* a_2 .* (1 - a_2);
  
  % remove error term for bias unit from the hidden layer
  d_2 = d_2(2:end);
 
  % accumulate gradient terms for this example
  Theta2_grad = Theta2_grad + d_3 * a_2';
  Theta1_grad = Theta1_grad + d_2 * a_1';
  
endfor

% weight gradient terms by example size
Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad; 

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) += (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += (lambda / m) * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
