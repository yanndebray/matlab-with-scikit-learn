function kw = pyargs(varargin)
%PYARGS  Keyword arguments for a Python call, as in MATLAB.
%   py.call(cmp, 'render_all', preds, pyargs('output_dir', site_dir))
%
%   Names must be valid identifiers. The result is a PyDict flagged as
%   keyword arguments; py.call splats it into **kwargs.

    if mod(numel(varargin), 2) ~= 0
        error('py:pyargs:pairs', 'pyargs takes name/value pairs');
    end
    kw = PyDict(true);
    for i = 1:2:numel(varargin)
        name = varargin{i};
        if ~ischar(name) && ~isa(name, 'string')
            error('py:pyargs:name', 'keyword names must be text');
        end
        kw = insert(kw, char(name), varargin{i + 1});
    end
end
