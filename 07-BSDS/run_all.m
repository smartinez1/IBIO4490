addpath(genpath(pwd))
clear all;close all;clc;

DIR = 'ownSegmentation_finalexpKM/'; % cambiar con lo que se quiere

dir_ctrl = 'train';% grupo que se realizo las pruebas 'train' 'test

SEGDIR = 'data/segs/';% direccion de los datos a tomar "data modificado"

segDir = char(strcat(SEGDIR,DIR));
DIR = split(DIR,'_');
outDir = char(strcat(SEGDIR,'segs_',DIR(2)));
%%
seg2uintseg(segDir,outDir);
%%
imgDir = strcat('data/images/',dir_ctrl);
gtDir = strcat('data/groundTruth/',dir_ctrl);

inDir = char(strcat(SEGDIR,'segs_',DIR(2))); %Nosotros
outDir = char(strcat('data/eval/test_',dir_ctrl,'_',DIR(2)));% Nosotros

%inDir = char(strcat('data/ucm2/',dir_ctrl)); % profe
%outDir = char(strcat('data/eval/test_prof_',dir_ctrl)); % profe
mkdir(outDir);
aux = dir(inDir);
segs = load(fullfile(aux(3).folder,aux(4).name));
nthresh = numel(segs.segs);
%nthresh = 9
clear segs;
%%
addpath(genpath(pwd))
tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

%%
%plot de toda las imagenes
plot_eval(char(strcat('data/eval/test_prof_',dir_ctrl)),'b'); % profe
plot_eval('data/eval/test_train_GMMlab','r')
plot_eval('data/eval/test_train_justrgbgmm','c')
plot_eval('data/eval/test_train_labOhneRaumlicheInformation','m')
plot_eval('data/eval/test_train_rgbgmm','k')

plot_eval('data/eval/test_prof_test','b')
plot_eval('data/eval/test_test_finalexpGMM','r')
plot_eval('data/eval/test_test_finalexpKM','y')


