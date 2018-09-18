folder = 'C:\Users\tbureete\OneDrive\Purdue\2018\Fall\ABE591\lab3\lab3images';
filePattern = fullfile(folder, '*.tif');
srcFiles = dir(filePattern);
numImages = size(srcFiles,1);

k = 3; 
fullFileName = fullfile(srcFiles(k).folder, srcFiles(k).name);
rgbImage = imread(fullFileName);
for i = 1 : 3
    subplot(2, 4, i);
    imshow(rgbImage(:,:,i))
    [pixelCount, grayLevels] = imhist(rgbImage(:,:,i));
    subplot(2, 4, i+4);
    bar(pixelCount);
    xlim([0 grayLevels(end)]); % Scale x axis manually.
    
end
