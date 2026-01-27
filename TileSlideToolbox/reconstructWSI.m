clear;clc;close all
samplePath = '.\TCGA-02-0069-01Z-00-DX1'
fileFolder = fullfile(samplePath, '*.png');
fileNameStruct = dir(fileFolder);
fileName = {fileNameStruct.name}';

fileNum = length(fileName);
col = zeros(fileNum, 1);
row = zeros(fileNum, 1);

for i = 1:fileNum
    t = imread(strcat(fileNameStruct(i).folder, "\", fileName{i}));
    fileSplit = strsplit(fileName{i}, {'_', '.'});
    col(i, 1) = str2double(fileSplit{2}) + 1;
    row(i, 1) = str2double(fileSplit{3}) + 1;
end

maxCol = max(col);
maxRow = max(row);
img = zeros(maxRow, maxCol);
for i = 1:fileNum
    img(int32(row(i, 1)), int32(col(i, 1))) = 1;
end

imshow(img)
