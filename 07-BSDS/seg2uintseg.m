function seg2uintseg(segDir,outDir)
disp('entrando a seg2uintseg\n\n\n\n')
mkdir(outDir);
addpath(outDir)
S= dir(fullfile(segDir,'*.mat'));
tic;
for i =1:numel(S),
outFile = fullfile(outDir,[S(i).name(1:end-4) '.mat']);
if exist(outFile,'file'), continue; end
segFile=fullfile(segDir,S(i).name);
load(segFile)
u = {};
for k = 1:numel(segs)
u{end +1} = uint16(segs{k});
end
segs=u;
save(outFile,'segs');
end
toc;