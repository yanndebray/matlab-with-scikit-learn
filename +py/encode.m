function [enc, ctx] = encode(value, ctx)
%ENCODE  MATLAB value -> bridge wire value (see pybridge.py for the format).
%   [ENC, CTX] = py.encode(VALUE, CTX) returns the JSON-ready struct and the
%   updated call context (numeric data is written to binary blobs in
%   CTX.dir, so CTX has to be threaded through).
%
%   Mapping:
%       char / string scalar   -> str
%       finite numeric scalar  -> float
%       numeric array          -> numpy array (float64, vectors squeezed to 1-D)
%       logical                -> bool / numpy bool array
%       cell                   -> list
%       scalar struct          -> dict, keyed by field name
%       PyDict                 -> dict
%       PyObject               -> module reference, handle, or bound attribute

    % RunMat's accelerate layer can hand back a gpuArray from ordinary
    % arithmetic; gather() brings it home (double() does not). See RUNMAT.md.
    if isa(value, 'gpuArray')
        value = gather(value);
    end

    if isa(value, 'PyObject')
        if strcmp(kindof(value), 'matrix')
            data = payload(value);
            if islogical(data)
                [enc, ctx] = py.blob(uint8(data), 'uint8', 'u1', size(data), ctx, false);
            else
                [enc, ctx] = py.blob(double(data), 'double', 'f8', size(data), ctx, false);
            end
        else
            enc = pyref(value);
        end
        return;
    end

    if isa(value, 'PyDict')
        ks = keys(value);
        vs = vals(value);
        items = cell(1, numel(ks));
        for i = 1:numel(ks)
            pair = struct();
            [pair.k, ctx] = py.encode(ks{i}, ctx);
            [pair.v, ctx] = py.encode(vs{i}, ctx);
            items{i} = pair;
        end
        enc = struct();
        enc.pytag = 'dict';
        enc.items = items;
        return;
    end

    if ischar(value)
        if ~isempty(value) && size(value, 1) > 1
            error('py:unsupported', ...
                'only single-row char arrays convert to str (got %dx%d)', ...
                size(value, 1), size(value, 2));
        end
        enc = struct();
        enc.pytag = 'str';
        enc.v = value;
        return;
    end

    if isa(value, 'string')
        if numel(value) ~= 1
            [enc, ctx] = py.encode(num2cell(value), ctx);
            return;
        end
        enc = struct();
        enc.pytag = 'str';
        enc.v = char(value);
        return;
    end

    if iscell(value)
        items = cell(1, numel(value));
        for i = 1:numel(value)
            [items{i}, ctx] = py.encode(value{i}, ctx);
        end
        enc = struct();
        enc.pytag = 'list';
        enc.items = items;
        return;
    end

    if isa(value, 'struct')
        if numel(value) ~= 1
            error('py:unsupported', ...
                'struct arrays do not convert; pass a cell of structs instead');
        end
        names = fieldnames(value);
        items = cell(1, numel(names));
        for i = 1:numel(names)
            pair = struct();
            [pair.k, ctx] = py.encode(names{i}, ctx);
            [pair.v, ctx] = py.encode(value.(names{i}), ctx);
            items{i} = pair;
        end
        enc = struct();
        enc.pytag = 'dict';
        enc.items = items;
        return;
    end

    if islogical(value)
        if numel(value) == 1
            enc = struct();
            enc.pytag = 'bool';
            enc.v = value;
            return;
        end
        [enc, ctx] = py.blob(uint8(value), 'uint8', 'u1', size(value), ctx, true);
        return;
    end

    if isnumeric(value)
        if numel(value) == 1 && isfinite(value)
            enc = struct();
            enc.pytag = 'num';
            enc.v = double(value);
            return;
        end
        % Non-finite scalars and every array go through a binary blob: JSON
        % has no NaN/Inf, and blobs are how big matrices stay cheap.
        [enc, ctx] = py.blob(double(value), 'double', 'f8', size(value), ctx, true);
        return;
    end

    error('py:unsupported', 'cannot send a %s to Python', class(value));
end
