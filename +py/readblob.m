function out = readblob(enc, ctx)
%READBLOB  Read a binary blob the Python side wrote, as a MATLAB matrix.
%   Internal helper for py.decode; separate file because a RunMat package
%   file may only hold one function. See RUNMAT.md.
    path = fullfile(ctx.dir, char(enc.file));
    fid = fopen(path, 'rb');
    if fid < 0
        error('py:blobMissing', 'the Python side promised %s but it is gone', path);
    end
    islogical_ = strcmp(char(enc.dtype), 'u1');
    if islogical_
        raw = fread(fid, Inf, 'uint8');
    else
        raw = fread(fid, Inf, 'double');
    end
    fclose(fid);

    % Index the field's value, not the field: RunMat lowers enc.shape(:)
    % into a getfield() with a ':' index and rejects it. See RUNMAT.md.
    shape = double(enc.shape);
    shape = shape(:)';
    if numel(shape) < 2
        shape = [1 numel(raw)];
    end
    out = reshape(raw, shape);
    if islogical_
        out = logical(out);
    end
    % Keep bridge results host-side. RunMat's accelerate layer moves arrays
    % of this size onto the device, and a device-resident array cannot be
    % compared with == against a host one. See RUNMAT.md.
    out = gather(out);
end
