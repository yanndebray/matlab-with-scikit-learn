function ctx = cleanup(ctx)
%CLEANUP  Drop a call's scratch directory unless RUNMAT_PY_KEEP=1.
    if ~ctx.keep && exist(ctx.dir, 'dir') == 7
        rmdir(ctx.dir, 's');
    end
end
