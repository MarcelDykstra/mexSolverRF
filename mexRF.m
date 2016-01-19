classdef mexRF < handle
  methods
    function self = mexRF(A, L, U, p, q)
      warn = warning('off', 'all');
      loadlibrary('mex_rf', @mex_rf_proto);
      warning(warn);
      calllib('mex_rf', 'mexRfInitialize', A', L', U', p', q');
    end
  end
end
