function mod = import_module(name)
%IMPORT_MODULE  Import a Python module and return a handle to it.
%   CMP = py.importlib.import_module('compare') mirrors MATLAB's
%   py.importlib.import_module. The import happens in Python straight away,
%   so a missing module or a syntax error in it surfaces here rather than at
%   the first call. The repo root and the current directory are on sys.path.

    ctx = py.session('call');

    request = struct();
    request.op = 'import';
    request.name = name;

    response = py.raw(request, ctx);
    mod = py.decode(response.value, ctx);
    py.cleanup(ctx);
end
