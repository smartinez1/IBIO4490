data_path = ['../data/'];
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces');
image_files = dir( fullfile(train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
i=randperm(length(image_files),9);

cont=1;
figure
for idx=i
im = single(imread(fullfile(train_path_pos,image_files(idx).name)))/255;
subplot(3,3,cont)
imshow(im)
cont=cont+1;
end

data_path = ['../data/'];
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes');;
image_files_1 = dir( fullfile( non_face_scn_path, '*.jpg' )); %Caltech Faces stored as .jpg
in=randperm(length(image_files_1),9);

cont=1;
figure
for idx=in
im = (imread(fullfile(non_face_scn_path,image_files_1(idx).name)));
subplot(3,3,cont)
imshow(im)
cont=cont+1;
end