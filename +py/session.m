function out = session(action)
%SESSION  Interop session state for the RunMat py.* bridge.
%   S = py.session() returns the session as a struct, creating its scratch
%   directory on first use:
%       home     directory holding pybridge.py and +py
%       python   Python executable      (override: RUNMAT_PYTHON)
%       bridge   path to pybridge.py
%       root     scratch directory for this session
%       objects  where pickled Python handles live
%       verbose  echo the Python side's stdout  (RUNMAT_PY_VERBOSE=0 to mute)
%       keep     keep per-call scratch dirs     (RUNMAT_PY_KEEP=1)
%
%   CTX = py.session('call') also creates a fresh per-call directory and
%   returns it as CTX.dir, ready to hand to py.encode / py.decode.
%
%   py.session('reset') removes the scratch directory and forgets the
%   session, which invalidates every handle it was holding.
%
%   Two RunMat quirks shape this function (see RUNMAT.md): field access on a
%   persistent struct lowers to a graphics get() and errors out, so the state
%   is one persistent variable per field; and persistent variables start life
%   as 0 rather than [], so the "not initialised yet" test is ~ischar(home).

    persistent home python bridge root objects verbose keep calls

    if nargin < 1
        action = 'get';
    end

    if strcmp(action, 'reset')
        if ischar(root) && exist(root, 'dir') == 7
            rmdir(root, 's');
        end
        home = 0;
        root = 0;
        out = [];
        return;
    end

    if ~ischar(home)
        home = fileparts(fileparts(mfilename('fullpath')));  % strip /+py

        bridge = fullfile(home, 'pybridge.py');
        % RunMat reports a .py file as 3, MATLAB as 2 — only 0 means missing.
        if exist(bridge, 'file') == 0
            error('py:bridgeMissing', ...
                'pybridge.py not found next to +py (looked in %s)', home);
        end

        python = getenv('RUNMAT_PYTHON');
        if isempty(python) || ~ischar(python)
            python = 'python3';
        end

        verbose = double(~strcmp(getenv('RUNMAT_PY_VERBOSE'), '0'));
        keep = double(strcmp(getenv('RUNMAT_PY_KEEP'), '1'));

        [~, tag] = fileparts(tempname);
        root = fullfile(tempdir, ['runmat-py-' tag]);
        objects = fullfile(root, 'objects');
        mkdir(root);
        mkdir(objects);
        calls = 0;
    end

    out = struct();
    out.home = home;
    out.python = python;
    out.bridge = bridge;
    out.root = root;
    out.objects = objects;
    out.verbose = logical(verbose);
    out.keep = logical(keep);

    if strcmp(action, 'call')
        calls = calls + 1;
        out.dir = fullfile(root, sprintf('call%d', calls));
        mkdir(out.dir);
        out.blobs = 0;
    end
end
