clear all; clc;
curr_path = pwd; cd ..; addpath(pwd); cd(curr_path);

load matrix;

[L, U, p, q] = lu(A, 'vector');

objRF = mexRF(A, L, U, p, q);

