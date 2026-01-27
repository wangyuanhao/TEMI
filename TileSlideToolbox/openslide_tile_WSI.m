function [tile_status, validTile] = openslide_tile_WSI(WSI, tileSize, magnification, overlap, outdir, filterdir)
%% Input:
%   WSI: filename of SVS 
%   tileSize: size of each tile, tileSize x tileSize
%   magnification: manification
%   overlap: overlap of each tile
%   outdir: directory to save tiles
%   filterdir: directory to save invalid tiles

% Parameter
Bkg = 50;
ValueThreshd = 220;


% Load openslide library
openslide_load_library();
overlap_factor = 1 - overlap;
% Open whole-slide image
slidePtr = openslide_open(WSI);

% Get whole-slide image properties
[mppX, mppY, width, height, numberOfLevels, ...
    downsampleFactors, objectivePower] = openslide_get_slide_properties(slidePtr);

if objectivePower < magnification
    tile_status = 0;
    return
end

%% Original
levelToUse = find(objectivePower./round(downsampleFactors) >= magnification,1) - 1;
% Determine which level equal to or greater than desired magnification/objectivePower
if numberOfLevels > 0
    for level = numberOfLevels:-1:1
     if objectivePower / round(downsampleFactors(level)) < magnification
         continue
     end
     levelToUse = level - 1;
    end
else
    if (objectivePower == magnification)
        levelToUse = 0;
    else
        tile_status = 0;
        return
    end
end


% Adjust image width and height accordingly
width = round(width / downsampleFactors(levelToUse + 1));
height = round(height / downsampleFactors(levelToUse + 1));

% Read all tiles
validTile = 1;
for colInd = 0 : floor(width/tileSize)
    for rowInd = 0 : floor(width/tileSize)
        % Set start indices and size
        startX = colInd * ceil(tileSize * overlap_factor);
        startY = rowInd * ceil(tileSize * overlap_factor);
        widthX = min(tileSize,tileSize - ((colInd + 1) * tileSize - width));
        heightY = min(tileSize,tileSize - ((rowInd + 1) * tileSize - height));
        
        if widthX <= 0 || heightY <= 0
            continue
        end

        % Read the tile
        ARGB = openslide_read_region(slidePtr,...
                                     startX,startY,widthX,heightY,...
                                     levelToUse);
       
        rgbImage = ARGB(:,:,2:4);
        % need to be squared image
        if size(rgbImage,1) == size(rgbImage, 2)
            filterFlag = filter_tile(rgbImage, Bkg, ValueThreshd);
            outfi = sprintf('%d_%d.png', rowInd+1, colInd+1);
            if filterFlag == 0 % a valid tile
                % save tile as png       
                imwrite(rgbImage,strcat(outdir, outfi)) 
                validTile = validTile + 1;
            else
                % a non-valid tile
                imwrite(rgbImage,strcat(filterdir, outfi)) 
            end
        end
        
    end
end

% Close whole-slide image, note that the slidePtr must be removed manually
openslide_close(slidePtr)
clear slidePtr

% Unload library
openslide_unload_library
tile_status = 1;
end


function [filterFlag] = filter_tile(rgbImage, Bkg, ValueThreshd)
% ref https://github.com/ncoudray/DeepPATH/blob/master/DeepPATH_code/00_preprocessing/0b_tileLoop_deepzoom4.py
    grayImage = rgb2gray(rgbImage);
    BW = (grayImage > ValueThreshd);
    avgBkg = mean(BW(:));
    if avgBkg <= (Bkg / 100.0)
        filterFlag = 0;
    else
        filterFlag = 1;
    end
 
end