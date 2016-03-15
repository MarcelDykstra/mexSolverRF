clear all; clc;
curr_path = pwd; cd ..; addpath(pwd); cd(curr_path);

load matrix;
A = repmat({A}, 1, 5);
A = blkdiag(A{:});
num_rows = size(A, 1);
b = ones(num_rows, 1);

rfA = mexRF(A);

tic_rf = tic;
rfA.refactor(A .* 1.2);
x = rfA \ b;
disp(['cuSolverRF: ' num2str(toc(tic_rf)) 's']);

tic_mat = tic;
xm = A \ b;
disp(['Matlab: ' num2str(toc(tic_mat)) 's']);

figure;
plot(x); hold on; plot(xm);
axis tight; box on; grid on;
