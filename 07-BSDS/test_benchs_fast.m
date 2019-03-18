%% Examples of benchmarks for different input formats
addpath benchmarks
clear all;close all;clc;

%% 2.   morphological version for :boundary benchmark for results stored as contour images

imgDir = 'data/images';
gtDir = 'data/groundTruth';
pbDir = 'data/png';
outDir = 'eval/test_bdry_fast';
mkdir(outDir);
nthresh = 99;

tic;
boundaryBench_fast(imgDir, gtDir, pbDir, outDir, nthresh);
toc;


%% 4. morphological version for : all the benchmarks for results stored as a cell of segmentations
imgDir = 'data/images';
gtDir = 'data/groundTruth';

inDir = 'data/segs_GMMLab';% yo
outDir = 'eval/test_all_fast_mio';% yo

inDir = 'data/ucm2'; % profe
outDir_prof = 'eval/test_all_fast'; % profe

mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir_prof,'r');
plot_eval(outDir,'r');

