function out = matrix(value)
%MATRIX  Send a vector to Python as a 2-D array instead of a 1-D one.
%   Because MATLAB has no 1-D array, the bridge collapses 1-by-N and N-by-1
%   data to a 1-D numpy array — which is what sklearn wants for labels.
%   Wrap a vector in py.matrix to keep both dimensions:
%
%       py.call(np, 'shape', py.matrix([1 2 3]))   % (1, 3), not (3,)

    if ~isnumeric(value) && ~islogical(value)
        error('py:matrix:unsupported', ...
            'py.matrix expects numeric or logical data, got %s', class(value));
    end
    out = PyObject('matrix', value);
end
