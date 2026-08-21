function value = exec(code, names, output)
%EXEC  Run Python statements, optionally returning one variable afterwards.
%   py.exec('import sys; print(sys.version)')
%   v = py.exec('y = x * 2', struct('x', [1 2 3]), 'y')

    if nargin < 2
        names = struct();
    end

    ctx = py.session('call');

    request = struct();
    request.op = 'eval';
    request.code = code;
    request.statements = true;
    if nargin >= 3
        request.output = output;
    end
    request.names = struct();
    fields = fieldnames(names);
    for i = 1:numel(fields)
        [request.names.(fields{i}), ctx] = py.encode(names.(fields{i}), ctx);
    end

    response = py.raw(request, ctx);
    value = py.decode(response.value, ctx);
    py.cleanup(ctx);
end
