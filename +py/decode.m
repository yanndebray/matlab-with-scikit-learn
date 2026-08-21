function value = decode(enc, ctx)
%DECODE  Bridge wire value -> MATLAB value (inverse of py.encode).
%   Mapping:
%       None            -> []
%       float / int     -> double
%       str             -> char
%       bool            -> logical
%       numpy array     -> double (or logical) matrix
%       list / tuple    -> cell
%       dict            -> PyDict
%       module / object -> PyObject
%
%   Written as an if/elseif chain rather than a switch: RunMat cannot switch
%   on char values at all. See RUNMAT.md.

    if isempty(enc) || ~isa(enc, 'struct') || ~isfield(enc, 'pytag')
        error('py:badWire', 'response was not a tagged value');
    end
    tag = char(enc.pytag);

    if strcmp(tag, 'none')
        value = [];

    elseif strcmp(tag, 'num')
        if isfield(enc, 'special')
            special = char(enc.special);
            if strcmp(special, 'nan')
                value = NaN;
            elseif strcmp(special, 'inf')
                value = Inf;
            elseif strcmp(special, '-inf')
                value = -Inf;
            else
                error('py:badWire', 'unknown float marker %s', special);
            end
        else
            value = double(enc.v);
        end

    elseif strcmp(tag, 'str')
        value = char(enc.v);

    elseif strcmp(tag, 'bool')
        value = logical(enc.v);

    elseif strcmp(tag, 'array')
        value = py.readblob(enc, ctx);

    elseif strcmp(tag, 'list') || strcmp(tag, 'tuple')
        items = py.aslist(enc.items);
        value = cell(1, numel(items));
        for i = 1:numel(items)
            value{i} = py.decode(items{i}, ctx);
        end

    elseif strcmp(tag, 'dict')
        items = py.aslist(enc.items);
        value = PyDict();
        for i = 1:numel(items)
            value = insert(value, ...
                py.decode(items{i}.k, ctx), ...
                py.decode(items{i}.v, ctx));
        end

    elseif strcmp(tag, 'module') || strcmp(tag, 'handle')
        value = PyObject(enc);

    elseif strcmp(tag, 'callable')
        % Only py.getattr sees this; it rewraps as a bound PyObject.
        value = enc;

    else
        error('py:badWire', 'unknown pytag %s', tag);
    end
end
