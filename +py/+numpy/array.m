function out = array(value)
%ARRAY  numpy.array, as far as the bridge is concerned.
%   The bridge already marshals MATLAB numeric and logical data as numpy
%   arrays, so this is a pass-through that exists for source compatibility
%   with MATLAB's py.numpy.array(x). Wrapping every prediction vector in a
%   real Python round trip would cost an interpreter launch per call and
%   hand back a handle that the next call would only unpickle again.
%
%   Use py.matrix(x) if you need a vector to stay 2-D on the Python side.

    if ~isnumeric(value) && ~islogical(value)
        error('py:numpy:unsupported', ...
            'py.numpy.array expects numeric or logical data, got %s', class(value));
    end
    out = value;
end
