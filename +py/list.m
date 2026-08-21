function out = list(value)
%LIST  Convert a cell array to a Python list.
%   Cells already marshal as lists; this is here for source compatibility.
    if nargin < 1
        out = {};
    elseif iscell(value)
        out = value;
    else
        out = num2cell(value);
    end
end
