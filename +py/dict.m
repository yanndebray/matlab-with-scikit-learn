function d = dict(varargin)
%DICT  A Python dict you fill in on the MATLAB side.
%   D = py.dict() returns an empty PyDict. Assign into it with either
%   brace or paren indexing, the way the MATLAB driver does:
%
%       D = py.dict();
%       D{'fitctree'} = yhat;
%       D('TreeBagger') = other;
%
%   The dict is built locally and converted when it is passed to Python, so
%   filling it in costs nothing.

    d = PyDict();
    if mod(numel(varargin), 2) ~= 0
        error('py:dict:pairs', 'py.dict takes key/value pairs');
    end
    for i = 1:2:numel(varargin)
        d = insert(d, varargin{i}, varargin{i + 1});
    end
end
