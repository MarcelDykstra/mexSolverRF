clear all; clc;
curr_path = pwd; cd ..; addpath(pwd); cd(curr_path);

load matrix;

objRF = mexRF(A);
objRF.refactor(A);

