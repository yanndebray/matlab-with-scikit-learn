function [passed, failed] = check(name, condition, passed, failed)
%CHECK  Report one test result and tally it.
%   Lives in its own file because test_pyinterop.m is a script, and RunMat
%   resolves a script's local functions less predictably than plain files.
    if condition
        fprintf('  ok    %s\n', name);
        passed = passed + 1;
    else
        fprintf('  FAIL  %s\n', name);
        failed = failed + 1;
    end
end
