function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
nC=[0.01 0.03 0.1 0.3 1 3 10 30];
nsigma=[0.01 0.03 0.1 0.3 1 3 10 30];
% nC=[0.01 1];
% nsigma=[0.01 1];
temp=1;
for i=1:length(nC)
    for j=1:length(nsigma)
        model = svmTrain(X, y, nC(i), @(x1, x2)gaussianKernel(x1, x2, nsigma(j)));
        pred  = svmPredict(model, Xval);
        error((i-1)*length(nC)+j) = mean(double(pred~= yval));
    end
end
[Terror,idx]=min(error);
cidx=floor(idx/length(nC))+1;
sigmaidx=idx-(cidx-1)*length(nC);

C=nC(cidx);
sigma=nsigma(sigmaidx);

        






% =========================================================================

end
