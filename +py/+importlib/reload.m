function mod = reload(mod)
%RELOAD  Re-import a module, picking up edits made since it was first loaded.
%   Kept for source compatibility with the MATLAB driver. Each bridge call
%   runs in a fresh interpreter, so modules are never stale in practice —
%   this still round-trips to Python so an edit that broke the module fails
%   loudly here.

    ctx = py.session('call');

    request = struct();
    request.op = 'reload';
    [request.target, ctx] = py.encode(mod, ctx);

    response = py.raw(request, ctx);
    mod = py.decode(response.value, ctx);
    py.cleanup(ctx);
end
