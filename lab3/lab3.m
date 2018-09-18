folder = 'C:\Users\tbureete\OneDrive\Purdue\2018\Fall\ABE591\lab3\lab3images';
filePattern = fullfile(folder, '*.tif');
srcFiles = dir(filePattern);
numImages = size(srcFiles,1);
fontSize = 12;

for k = 1 : numImages
    fullFileName = fullfile(srcFiles(k).folder, srcFiles(k).name);
    rgbImage = imread(fullFileName);
    redImage = rgbImage(:, :, 1);
    greenImage = rgbImage(:, :, 2);
    blueImage = rgbImage(:, :, 3);
    figure;
    set(gcf, 'Position', get(0, 'ScreenSize'));
    subplot(3, 4, 1);
    imshow(rgbImage);
    title('RGB Image', 'FontSize', fontSize);
    subplot(3, 4, 2);
    imshow(redImage);
    title('Red Image', 'FontSize', fontSize);
    subplot(3, 4, 3);
    imshow(greenImage);
    title('Green Image', 'FontSize', fontSize);
    subplot(3, 4, 4);
    imshow(blueImage);
    title('Blue Image', 'FontSize', fontSize);
end
