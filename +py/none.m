function out = none()
%NONE  Python None, for the rare API that distinguishes it from [].
%   py.encode sends MATLAB [] as an empty numpy array, matching MATLAB's own
%   interop; use py.none() when the callee really wants None.
    out = PyObject('none');
end
