function yhat = knnPredict(X_train, y_train, X_test, k)
%KNNPREDICT  Majority-vote k-nearest-neighbour classifier, in plain MATLAB.
%   RunMat has no fitcknn, so the port supplies one: standardise the features
%   the way sklearn's StandardScaler + KNeighborsClassifier pipeline does,
%   then take the modal label of the k nearest training points.
%
%   Ties go to the smaller label, which is what mode() does and close enough
%   to sklearn's behaviour for a benchmark.

    mu = mean(X_train, 1);
    sigma = std(X_train, 0, 1);
    sigma = sigma + (sigma == 0);   % guard constant columns. sigma(mask) = 1
                                    % would be the MATLAB idiom, but RunMat
                                    % cannot assign through a logical mask.

    Ztrain = (X_train - mu) ./ sigma;
    Ztest = (X_test - mu) ./ sigma;

    D = pdist2(Ztest, Ztrain);
    labels = y_train(:);
    yhat = zeros(size(Ztest, 1), 1);
    for i = 1:size(D, 1)
        [~, order] = sort(D(i, :));
        yhat(i) = mode(labels(order(1:k)));
    end

    % RunMat's accelerate layer may hand back a gpuArray; gather() brings it
    % home. double() does not — it keeps the gpuArray class. See RUNMAT.md.
    yhat = gather(yhat);
end
