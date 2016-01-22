classdef mexRF < handle
  methods
    function self = mexRF(A)
      warn = warning('off', 'all');
      loadlibrary('mex_rf', @mex_rf_proto);
      warning(warn);
      [L, U, p, q] = lu(A, 'vector');
      calllib('mex_rf', 'mexRfInitialize', A', L', U', p', q');
    end
    function delete(self)
      calllib('mex_rf','mexRfDestroy');
      unloadlibrary('mex_rf');
    end
  end
end
