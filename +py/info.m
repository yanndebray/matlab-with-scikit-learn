function out = info()
%INFO  What the bridge is talking to — the py.* stand-in for pyenv.
%   py.info() with no output prints the interpreter and the versions of the
%   packages this benchmark needs; with an output it returns them as a
%   struct. This is also the cheapest way to check the bridge works at all.

    ctx = py.session('call');

    request = struct();
    request.op = 'ping';
    response = py.raw(request, ctx);
    ping = py.decode(response.value, ctx);
    py.cleanup(ctx);

    facts = struct();
    names = keys(ping);
    for i = 1:numel(names)
        facts.(names{i}) = get(ping, names{i});
    end

    if nargout > 0
        out = facts;
        return;
    end

    fprintf('       bridge: %s\n', ctx.bridge);
    fprintf('   executable: %s\n', facts.executable);
    fprintf('      version: %s\n', facts.version);
    fprintf('       prefix: %s\n', facts.prefix);
    for name = {'numpy', 'sklearn', 'matplotlib', 'skore'}
        value = facts.(name{1});
        if isempty(value)
            value = '<not installed>';
        end
        fprintf('%13s: %s\n', name{1}, value);
    end
end
