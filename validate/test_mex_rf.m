clear all; clc;
curr_path = pwd; cd ..; addpath(pwd); cd(curr_path);

load matrix; num_rows = size(A, 1);

rfA = mexRF(A);
rfA.refactor(A);

b = ones(num_rows, 1);
x = rfA \ b;

xm = A \ b;

plot(x); hold on; plot(xm);
