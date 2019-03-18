addpath(genpath(pwd))
% Plot de las imagenes
% plot imagenes de train/ pruebas realizadas
plot_eval(char(strcat('data/eval/test_prof_train')),'b'); % profe
close all;
plot_eval('data/eval/test_train_GMMlab','r')
close all;
plot_eval('data/eval/test_train_justrgbgmm','c')
close all;
plot_eval('data/eval/test_train_labOhneRaumlicheInformation','m')
close all;
plot_eval('data/eval/test_train_rgbgmm','k')
pause();
delete('isoF.fig')
%%
% plor imagenes test 
plot_eval('data/eval/test_prof_test','b')
close all;
plot_eval('data/eval/test_test_finalexpGMM','r')
close all;
plot_eval('data/eval/test_test_finalexpKM','y')
pause();
delete('isoF.fig')
