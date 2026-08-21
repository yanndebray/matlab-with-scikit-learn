function value = getattr(target, name)
%GETATTR  Read an attribute of a Python object.
%   V = py.getattr(TARGET, NAME) converts data attributes to MATLAB values.
%   If the attribute is callable, V is a bound PyObject you can call:
%
%       f = py.getattr(cmp, 'dataset_keys');   % or:  f = cmp.dataset_keys
%       names = f();
%
%   PyObject's subsref routes obj.name here, so cmp.dataset_keys works.

    ctx = py.session('call');

    request = struct();
    request.op = 'getattr';
    [request.target, ctx] = py.encode(target, ctx);
    request.attr = name;

    response = py.raw(request, ctx);
    value = py.decode(response.value, ctx);
    py.cleanup(ctx);

    if isa(value, 'struct') && isfield(value, 'pytag') && strcmp(char(value.pytag), 'callable')
        value = PyObject('bound', target, name);
    end
end
