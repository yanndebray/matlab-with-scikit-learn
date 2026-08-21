function varargout = call(target, attr, varargin)
%CALL  Call a Python function, method, or callable object.
%   R = py.call(TARGET, ATTR, ARG1, ARG2, ...) evaluates
%   TARGET.ATTR(ARG1, ...) in Python and converts the result back.
%   TARGET is a PyObject (module or object handle); pass ATTR = '' to call
%   TARGET itself. Keyword arguments come from pyargs:
%
%       idx = py.call(cmp, 'render_all', preds, pyargs('output_dir', dir));
%
%   This is the workhorse of the bridge. RunMat cannot dispatch
%   obj.method(args) through a user-defined subsref, so py.call is how the
%   port spells what MATLAB writes as cmp.render_all(preds, ...).
%   See RUNMAT.md.

    ctx = py.session('call');

    request = struct();
    request.op = 'call';
    [request.target, ctx] = py.encode(target, ctx);
    request.attr = attr;

    args = {};
    kwargs = struct();
    for i = 1:numel(varargin)
        item = varargin{i};
        if isa(item, 'PyDict') && iskwargs(item)
            ks = keys(item);
            vs = vals(item);
            for k = 1:numel(ks)
                name = ks{k};
                if ~ischar(name) || ~isvarname(name)
                    error('py:badKeyword', ...
                        'keyword names must be valid identifiers');
                end
                [kwargs.(name), ctx] = py.encode(vs{k}, ctx);
            end
        else
            [encoded, ctx] = py.encode(item, ctx);
            args{end+1} = encoded;
        end
    end
    request.args = args;
    request.kwargs = kwargs;

    response = py.raw(request, ctx);
    varargout{1} = py.decode(response.value, ctx);
    py.cleanup(ctx);
end
