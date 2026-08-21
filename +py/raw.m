function response = raw(request, ctx)
%RAW  Run one bridge request through pybridge.py and return the raw response.
%   REQUEST is a struct describing the operation (see pybridge.py for the
%   wire format); CTX comes from py.session('call'). The returned struct has
%   the response's `value` field still in wire form — py.decode turns it into
%   a MATLAB value. A Python-side failure is raised as a MATLAB error.

    request.call_dir = ctx.dir;
    request.object_dir = ctx.objects;
    request.sys_path = {ctx.home, pwd};

    req_path = fullfile(ctx.dir, 'request.json');
    resp_path = fullfile(ctx.dir, 'response.json');

    fid = fopen(req_path, 'w');
    if fid < 0
        error('py:scratchUnwritable', 'cannot write %s', req_path);
    end
    fprintf(fid, '%s', jsonencode(request));
    fclose(fid);

    cmd = sprintf('"%s" "%s" "%s" "%s" 2>&1', ...
        ctx.python, ctx.bridge, req_path, resp_path);
    [~, chatter] = system(cmd);
    if ctx.verbose && ~isempty(chatter)
        fprintf('%s', chatter);
    end

    % exist() reports a .json file as 3 in RunMat, 2 in MATLAB: only 0 is
    % "missing". RunMat's error() also leaves \n in the format string alone,
    % so messages are assembled with sprintf first. See RUNMAT.md.
    if exist(resp_path, 'file') == 0
        error('py:bridgeFailed', '%s', sprintf( ...
            'the Python bridge produced no response.\ncommand: %s\noutput:\n%s', ...
            cmd, chatter));
    end
    response = jsondecode(fileread(resp_path));

    if ~response.ok
        err = response.error;
        error('py:PythonError', '%s', sprintf('%s: %s\n%s', ...
            char(err.type), char(err.message), char(err.traceback)));
    end
end
