%% Demo
cd('code')
clear all
close all
clc

load('variables.mat')
run('vlfeat/toolbox/vl_setup')
test_scn_path=fullfile('../Images/','extra_test_scenes');
[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);
i=randperm(length(dir(fullfile('../Images/','extra_test_scenes','*.jpg'))),1);
wa=dir(fullfile('../Images/','extra_test_scenes','*.jpg'));

 cur_bboxes=bboxes;
 for j = i:i
        cur_test_image = imread( fullfile( test_scn_path,wa(j).name));
        imshow(cur_test_image);
        hold on;
        bb = cur_bboxes(j,:);
        plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'y:','linewidth',2);
 end



