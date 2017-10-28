
clear ; close all; clc

data = load('newtrain.txt');

V = data(:,[1,2,3,4,5,6,7]); y = data(:, 8);
YProb = [0.68,0.66,0.56,0.45,0.59,0.46,0.58];
NProb = [0.29,0.36,0.43,0.55,0.43,0.67,0.46];
X = zeros(rows(V),1);

for i = 1:rows(V)
  for j = 1:columns(V)
    if (V (i, j) == 1)
      X(i,1) =  X(i,1) + YProb(1,j);
    else
      X(i,1) = X(i,1) + NProb(1,j);
    endif
  endfor
endfor



fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotdata2(X, y);

hold on;

xlabel('Scores ')
ylabel('Male/Female')


legend('Male', 'Female')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

printf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

hold on;

xlabel('Scores ')
ylabel('Male/Female')


legend('Male', 'Female')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

[m, n] = size(X);

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);


[cost, grad] = costFunction(initial_theta, X, y);

%fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
%fprintf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

% Compute and display cost and gradient with non-zero theta
test_theta = [0;2.71];
%test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);

fprintf('\nCost at test theta: %f\n', cost);

fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  fminunc to obtain the optimal theta
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

  
prob = sigmoid([1 3.8] * theta);
fprintf(['1. For a student with scores 3.8 we predict a Likelihood to have masculine behaviour ' ...
         ' with a probability of %f or percentage of %f\n'], prob,prob*100);
      
p = predict(theta, X);


prob = sigmoid([1 2.88] * theta);
fprintf(['2. For a student with scores 3.8 we predict a Likelihood to have masculine behaviour ' ...
         ' with a probability of %f or percentage of %f\n'], prob,prob*100);
      

p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('\n');


