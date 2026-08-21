function acc = accuracy(yhat, ytruth)
%ACCURACY  Share of labels that match.
%   Both sides are gathered first: once an array is a few hundred elements
%   long RunMat's accelerate layer keeps it on the device, and == between a
%   device array and a host one fails outright. mean(a == b) is the MATLAB
%   one-liner this replaces. See RUNMAT.md.

    a = gather(yhat);
    b = gather(ytruth);
    a = a(:);
    b = b(:);
    if numel(a) ~= numel(b)
        error('runmat:accuracy:size', ...
            'got %d predictions for %d labels', numel(a), numel(b));
    end
    acc = sum(a == b) / numel(a);
end
