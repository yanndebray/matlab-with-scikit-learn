function [enc, ctx] = blob(data, precision, dtype, shape, ctx, squeeze_)
%BLOB  Write numeric data to a binary blob and return its wire reference.
%   Internal helper for py.encode. It lives in its own file because a RunMat
%   package file may only hold one function: a call to pkg.name dispatches to
%   the *last* function in the file, and siblings are invisible to each
%   other. See RUNMAT.md.
    name = sprintf('i%d.bin', ctx.blobs);
    ctx.blobs = ctx.blobs + 1;
    path = fullfile(ctx.dir, name);

    fid = fopen(path, 'wb');
    if fid < 0
        error('py:scratchUnwritable', 'cannot write %s', path);
    end
    fwrite(fid, data, precision);   % column-major, which is what numpy reads
    fclose(fid);

    enc = struct();
    enc.pytag = 'array';
    enc.file = name;
    enc.dtype = dtype;
    enc.shape = double(shape);
    enc.squeeze = squeeze_;
end
