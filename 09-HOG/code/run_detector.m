% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = ... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
num_images=length(test_scenes);
numWin=16;
step=5;
th=1.2;
visualizePositivesSlices=false;
%initialize these as empty and incrementally expand them.
bboxes=zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

for i=1:num_images
     
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
% Run Classifier
    
    [M,N]=size(img);
    wins={};
    C=min([M,N]);
    wins{1}=[C,C];
    auxG=-1;
    fac=1.5;
    for winC=2:numWin
        auxG=auxG+1;
        aux=wins{winC-1};
        auxM=aux(1);
        auxN=aux(2);
        
        if auxG==0
        wins{winC}=[floor(auxM),floor(auxN/fac)];
        elseif auxG==1
        wins{winC}=[floor(auxM/fac),floor(auxN*fac)];
        else
        auxG=-1;
        wins{winC}=[floor(auxM),floor(auxN/fac)];   
        end
        
    end
    

    cur_x_min=[];
    cur_y_min=[];
    cur_x_max=[];
    cur_y_max=[];
    conf=[];
    
    Detections=0;
    for j = 1:numWin
    aux=wins{j};
    alfa=aux(1);
    beta=aux(2);
        for k=1:step:(M-alfa)+1
            for l=1:step:(N-beta)+1
                auxIm1=img(k:k+alfa-1,l:l+beta-1);
                auxIm=single(imresize(auxIm1,[36,36]));
                HOG= vl_hog(auxIm, feature_params.hog_cell_size,'numOrientations', 9);
                v =HOG(:);
                classify=v'*w+b;            
                if classify>th
                    Detections=Detections+1;
                    cur_x_min(Detections)=k;
                    cur_y_min(Detections)=l;
                    cur_x_max(Detections)=k+alfa-1;
                    cur_y_max(Detections)=l+beta-1;
                    conf(Detections)=classify;
                    if visualizePositivesSlices
                    figure(4)
                    imagesc(auxIm1)
                    colormap('gray')
                    title(num2str(classify))
                    pause(0.01)
                    disp(['New Box:',num2str(k),'_',num2str(l),'_',num2str(k+alfa-1),'_',num2str(l+beta-1)])
                    end
                end

            end   
        end
    end
     
    cur_bboxes=zeros(Detections,4);
    for idx=1:Detections
        cur_bboxes(idx,:)=[cur_y_min(idx),cur_x_min(idx),cur_y_max(idx),cur_x_max(idx)];
    end
    
     cur_confidences=(conf*4-2)';
     cur_image_ids(1:Detections,1)={test_scenes(i).name};
    

    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    

    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    
    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
    
end






