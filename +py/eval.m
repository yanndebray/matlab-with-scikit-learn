function value = eval(code, names)
%EVAL  Evaluate a Python expression.
%   V = py.eval('1 + 1')
%   V = py.eval('x.mean()', struct('x', [1 2 3]))
%
%   NAMES is an optional scalar struct whose fields become local variables
%   in Python. Use py.exec for statements.

    if nargin < 2
        names = struct();
    end

    ctx = py.session('call');

    request = struct();
    request.op = 'eval';
    request.code = code;
    request.names = struct();
    fields = fieldnames(names);
    for i = 1:numel(fields)
        [request.names.(fields{i}), ctx] = py.encode(names.(fields{i}), ctx);
    end

    response = py.raw(request, ctx);
    value = py.decode(response.value, ctx);
    py.cleanup(ctx);
end
