clear;clc;close all
WSIDir = './SVS';
subDir = dir(WSIDir);
tileSize = 512;
magnification = 20;
overlap = 0;
outDir = './Slide-512px-20m/';
fprintf('Running......\n')
WSICount = 1;
for i=3:length(subDir)
    
    %% get file name and path
    subDir_ = subDir(i).name;
    subSubPath = fullfile(WSIDir, subDir_, '*.svs');
    fileDir = dir(subSubPath);
    fileName = fileDir.name;
    splitFileName = strsplit(fileName, '.');
    sampleID = splitFileName{1};
    outDir_ = fullfile(outDir, 'valid', sampleID);
    checkDir(outDir_)
    prefixOutDir_ = strcat(fullfile(outDir_, sampleID), '_');
    filterDir = fullfile(outDir, 'filter',sampleID);
    checkDir(filterDir)
    prefixFilterDir = strcat(fullfile(filterDir, sampleID), '_');
    filePath = fullfile(fileDir.folder, fileDir.name);
    %% start tiling
    [tile_status, validTile] = openslide_tile_WSI(filePath, tileSize, magnification, overlap, prefixOutDir_, prefixFilterDir);
    fprintf('Proccessed %s, get %d valid tiles\n', sampleID, validTile);
    if tile_status == 1
        WSICount = WSICount + 1;
        fprintf('Proccessed #WSI: %d/%d\n', WSICount, length(subDir)-2);
    end

end



function checkDir(dir_)
if ~exist(dir_, 'dir')
    mkdir(dir_)
end

end
