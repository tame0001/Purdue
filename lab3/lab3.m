tic;
clc; % Clear command window.
clearvars; % Get rid of variables from prior run of this m-file.

folder = 'C:\Users\tbureete\OneDrive\Purdue\2018\Fall\ABE591\lab3\lab3images';
filePattern = fullfile(folder, '*.tif');
srcFiles = dir(filePattern);
numImages = size(srcFiles,1);
fontSize = 12;
beanCount = 1;
classArray = cell([1 400]);
indexArray = zeros([1 400]);
areaArray = zeros([1 400]); 
perimeterArray = zeros([1 400]);
majorAxisArray = zeros([1 400]);
minorAxisArray = zeros([1 400]);
eccentricityArray = zeros([1 400]);
solidityArray = zeros([1 400]);
hueArray = zeros([1 400]);
intensityArray = zeros([1 400]);
elongationArray = zeros([1 400]);

for k = 1 : numImages
    % load image file one by one
    fullFileName = fullfile(srcFiles(k).folder, srcFiles(k).name);
    beanClass = srcFiles(k).name(1 : regexp(srcFiles(k).name, '(\d.tif)') - 1);
    imageSet = regexp(srcFiles(k).name, '(\d)', 'match');
    rgbImage = imread(fullFileName);
    redImage = rgbImage(:, :, 1);
    greenImage = rgbImage(:, :, 2);
    blueImage = rgbImage(:, :, 3);
%     figure; % plot image file in each color Image
    set(gcf, 'Position', get(0, 'ScreenSize'));
    subplot(4, 4, 1);
    imshow(rgbImage);
    title('RGB Image', 'FontSize', fontSize);
    subplot(4, 4, 2);
    imshow(redImage);
    title('Red Image', 'FontSize', fontSize);
    subplot(4, 4, 3);
    imshow(greenImage);
    title('Green Image', 'FontSize', fontSize);
    subplot(4, 4, 4);
    imshow(blueImage);
    title('Blue Image', 'FontSize', fontSize);
    
    hR = subplot(4, 4, 6);
	[countsR, grayLevelsR] = imhist(redImage);
	[maxGLValueR, maxGLRLocation] = max(countsR);
	maxCountR = max(countsR);
	bar(countsR, 'r');
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	title('Histogram of Red Image', 'FontSize', fontSize);
    
	% Compute and plot the green histogram.
	hG = subplot(4, 4, 7);
	[countsG, grayLevelsG] = imhist(greenImage);
    [maxGLValueG, maxGLGLocation] = max(countsG);
	maxCountG = max(countsG);
	bar(countsG, 'g', 'BarWidth', 0.95);
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	title('Histogram of Green Image', 'FontSize', fontSize);
	
	% Compute and plot the blue histogram.
	hB = subplot(4, 4, 8);
	[countsB, grayLevelsB] = imhist(blueImage);
    [maxGLValueB, maxGLBLocation] = max(countsB);
	maxCountB = max(countsB);
	bar(countsB, 'b');
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	title('Histogram of Blue Image', 'FontSize', fontSize);
    
	% Plot all 3 histograms in one plot.
	gRGB = subplot(4, 4, 5);
	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2);
	grid on;
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	hold on;
	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2);
	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2);
	title('Histogram of All Bands', 'FontSize', fontSize);
    
    % Set all axes to be the same width and height.
	% This makes it easier to compare them.
    maxCount = max([maxCountR,  maxCountG, maxCountB]) * 1.25;
	axis([hR hG hB gRGB], [0 255 0 maxCount]);
    
    % find thrsehold level
    subplot(4, 4, 6);
    hold on;
    redThreshold = graythresh(redImage) * 255;
    yAxisRangeValues = ylim;
	line([redThreshold, redThreshold], yAxisRangeValues, 'Color', 'black');
%     line([maxGLRLocation, maxGLRLocation], yAxisRangeValues, 'Color', 'black');
	text(redThreshold + 5, double(0.7 * yAxisRangeValues(2)), "Threshold");
%     text(maxGLRLocation + 5, double(0.9 * yAxisRangeValues(2)), "Peak");
    
    subplot(4, 4, 7);
    hold on;
    greenThreshold = graythresh(greenImage) * 255;
    yAxisRangeValues = ylim;
	line([greenThreshold, greenThreshold], yAxisRangeValues, 'Color', 'black');
%     line([maxGLGLocation, maxGLGLocation], yAxisRangeValues, 'Color', 'black');
	text(greenThreshold + 5, double(0.7 * yAxisRangeValues(2)), "Threshold");
%     text(maxGLGLocation + 5, double(0.9 * yAxisRangeValues(2)), "Peak");
    
    subplot(4, 4, 8);
    hold on;
    blueThreshold = graythresh(blueImage) * 255;
    yAxisRangeValues = ylim;
	line([blueThreshold, blueThreshold], yAxisRangeValues, 'Color', 'black');
%     line([maxGLBLocation, maxGLBLocation], yAxisRangeValues, 'Color', 'black');
	text(blueThreshold + 5, double(0.7 * yAxisRangeValues(2)), "Threshold");
%     text(maxGLBLocation + 5, double(0.9 * yAxisRangeValues(2)), "Peak");

    % Now apply each color band's particular thresholds to the color band
    if redThreshold < maxGLRLocation
        redMask = (redImage < redThreshold);
    else
        redMask = (redImage > redThreshold);
    end
    subplot(4, 4, 10);
    redMask = morphological(redMask);
	imshow(redMask .* rgbImage);
	title('Red Mask', 'FontSize', fontSize);
    
    if greenThreshold < maxGLGLocation
        greenMask = (greenImage < greenThreshold);
    else
        greenMask = (greenImage > greenThreshold);
    end
    subplot(4, 4, 11);
    greenMask = morphological(greenMask);
	imshow(greenMask .* rgbImage);
	title('Green Mask', 'FontSize', fontSize);
        
    if blueThreshold < maxGLBLocation
        blueMask = (blueImage < blueThreshold);
    else
        blueMask = (blueImage > blueThreshold);
    end
    subplot(4, 4, 12);
    blueMask = morphological(blueMask);
	imshow(blueMask .* rgbImage);
	title('Blue Mask', 'FontSize', fontSize);
    
    subplot(4, 4, 9);
    imshow(rgbImage);
    title('Original Image', 'FontSize', fontSize);
    
    subplot(4, 4, 13);
    blueScore = evaluateMask(blueThreshold, countsB);
%     greenScore = evaluateMask(greenThreshold, countsG);
    redScore = evaluateMask(redThreshold, countsR);
    if blueScore < redScore
        bestMask = blueMask;
    else
        bestMask = redMask;
    end
    maskedImage = rgbImage .* bestMask;
    imshow(maskedImage);
    title('Masked Image', 'FontSize', fontSize);
    
    labeledImage = bwlabel(bestMask);
    coloredLabels = label2rgb (labeledImage, 'hsv', 'k', 'shuffle');
    subplot(4, 4, 14);
    imshow(coloredLabels);
%     Make sure image is not artificially 
%     stretched because of screen's aspect ratio.
    axis image; 
    title("Labeled Image", 'FontSize', fontSize);
    
    beanMeasurements = regionprops(logical(bestMask), 'all');
    numberOfBeans = size(beanMeasurements, 1);
    boundaries = bwboundaries(bestMask);
    numberOfBoundaries = size(boundaries, 1);
    subplot(4, 4, 15);
    imshow(rgbImage);
    title('Outlines', 'FontSize', fontSize); 
    axis image; 
    hold on;
    for m = 1 : numberOfBoundaries
        thisBoundary = boundaries{m};
        plot(thisBoundary(:,2), thisBoundary(:,1));
    end
    hold off;
    
    for i = 1 : numberOfBeans
        classArray{beanCount} = beanClass;
        indexArray(beanCount) = 25 * (str2double(imageSet{1}) - 1) + i;
        areaArray(beanCount) = beanMeasurements(i).Area;
        perimeterArray(beanCount) = beanMeasurements(i).Perimeter;
        majorAxisArray(beanCount) = beanMeasurements(i).MajorAxisLength;
        minorAxisArray(beanCount) = beanMeasurements(i).MinorAxisLength;
        eccentricityArray(beanCount) = beanMeasurements(i).Eccentricity;
        solidityArray(beanCount) = beanMeasurements(i).Solidity;
        beanImage = imcrop(maskedImage,beanMeasurements(1).BoundingBox);
        beanImageGray = rgb2gray(beanImage);
        beanImage = double(beanImage) / 255; 
        R = beanImage(:, :, 1);
        G = beanImage(:, :, 2);
        B = beanImage(:, :, 3);
        hsv = rgb2hsv(beanImage);
        hue = hsv(:, :, 1);
        hueArray(beanCount) = mean(hue(hue>0));
        intensity = (R + G +B)./3;
        intensityArray(beanCount) = mean(intensity(intensity>0));
        d = 0;
        while(any(any(beanImageGray)))
            d = d + 1;
            beanImageGray = imerode(beanImageGray, strel('disk', 1));
        end
        elongationArray(beanCount) = beanMeasurements(i).Area / ((2*d)^2);
        beanCount = beanCount + 1;
    end
end

aspectRatioArray = majorAxisArray ./ minorAxisArray;
compactnessArray = (perimeterArray .^2) ./ areaArray;
rEqArray = (perimeterArray ./ (2 * pi)) + 0.5;
roundnessArray = areaArray ./ ((rEqArray .^2) .* pi);

varNames = {'Class', 'Index', 'Area', 'Hue', 'Intensity', 'Perimeter', ...
            'MajorAxis', 'MinorAxis', 'Eccentricity', 'Solidity', ...
            'Elongation', 'AspectRatio', 'Compactness', 'Roundness'};
featureTable = table(string(classArray'), ...
                     indexArray', ...
                     areaArray', ...
                     hueArray', ...
                     intensityArray', ...
                     perimeterArray', ...
                     majorAxisArray', ...
                     minorAxisArray', ...
                     eccentricityArray', ...
                     solidityArray', ...
                     elongationArray', ...
                     aspectRatioArray', ...
                     compactnessArray', ...
                     roundnessArray', ...
                     'VariableNames', varNames);
                 
classArray = unique(classArray);
varTypes = {'double', 'double', 'double', 'double', 'double', 'double', ...
            'double', 'double', 'double', 'double', 'double', 'double', ...
            'double', 'double', 'double', 'double', 'double', 'double', ...
            'double', 'double', 'double', 'double', 'double', 'double'};
varNames = {'AreaMean', 'AreaVariance', 'HueMean', 'HueVariance', ...
            'IntensityMean', 'IntensityVariance', 'PerimeterMean', ...
            'PerimeterVariance', 'MajorAxisMean', 'MajorAxisVariance', ...
            'MinorAxisMean', 'MinorAxisVariance', 'EccentricityMean', ...
            'EccentricityVariance', 'SolidityMean', 'SolidityVariance', ...
            'ElongationMean', 'ElongationVariance', 'AspectRatioMean', ...
            'AspectRatioVariance', 'CompactnessMean', ...
            'CompactnessVariance', 'RoundnessMean', 'RoundnessVariance'};
statsTable = table('Size', [8 24], ...
                    'VariableTypes', varTypes, ...
                    'VariableNames', varNames, ...
                    'RowNames', classArray);
classArray = string(classArray);

for i = 1:numel(classArray)
    className = char(classArray(i));
    
    u = mean(featureTable.Area(featureTable.Class == classArray(i)));
    statsTable(className, 'AreaMean') = {u};
    v = var(featureTable.Area(featureTable.Class == classArray(i)));
    statsTable(className, 'AreaVariance') = {v};
    
    u = mean(featureTable.Hue(featureTable.Class == classArray(i)));
    statsTable(className, 'HueMean') = {u};
    v = var(featureTable.Hue(featureTable.Class == classArray(i)));
    statsTable(className, 'HueVariance') = {v};
    
    u = mean(featureTable.Intensity(featureTable.Class == classArray(i)));
    statsTable(className, 'IntensityMean') = {u};
    v = var(featureTable.Intensity(featureTable.Class == classArray(i)));
    statsTable(className, 'IntensityVariance') = {v};
    
    u = mean(featureTable.Perimeter(featureTable.Class == classArray(i)));
    statsTable(className, 'PerimeterMean') = {u};
    v = var(featureTable.Perimeter(featureTable.Class == classArray(i)));
    statsTable(className, 'PerimeterVariance') = {v};
    
    u = mean(featureTable.MajorAxis(featureTable.Class == classArray(i)));
    statsTable(className, 'MajorAxisMean') = {u};
    v = var(featureTable.MajorAxis(featureTable.Class == classArray(i)));
    statsTable(className, 'MajorAxisVariance') = {v};
    
    u = mean(featureTable.MinorAxis(featureTable.Class == classArray(i)));
    statsTable(className, 'MinorAxisMean') = {u};
    v = var(featureTable.MinorAxis(featureTable.Class == classArray(i)));
    statsTable(className, 'MinorAxisVariance') = {v};
    
    u = mean(featureTable.Eccentricity(featureTable.Class == classArray(i)));
    statsTable(className, 'EccentricityMean') = {u};
    v = var(featureTable.Eccentricity(featureTable.Class == classArray(i)));
    statsTable(className, 'EccentricityVariance') = {v};
    
    u = mean(featureTable.Solidity(featureTable.Class == classArray(i)));
    statsTable(className, 'SolidityMean') = {u};
    v = var(featureTable.Solidity(featureTable.Class == classArray(i)));
    statsTable(className, 'SolidityVariance') = {v};
    
    u = mean(featureTable.Elongation(featureTable.Class == classArray(i)));
    statsTable(className, 'ElongationMean') = {u};
    v = var(featureTable.Elongation(featureTable.Class == classArray(i)));
    statsTable(className, 'ElongationVariance') = {v};
    
    u = mean(featureTable.AspectRatio(featureTable.Class == classArray(i)));
    statsTable(className, 'AspectRatioMean') = {u};
    v = var(featureTable.AspectRatio(featureTable.Class == classArray(i)));
    statsTable(className, 'AspectRatioVariance') = {v};
    
    u = mean(featureTable.Compactness(featureTable.Class == classArray(i)));
    statsTable(className, 'CompactnessMean') = {u};
    v = var(featureTable.Compactness(featureTable.Class == classArray(i)));
    statsTable(className, 'CompactnessVariance') = {v};
    
    u = mean(featureTable.Roundness(featureTable.Class == classArray(i)));
    statsTable(className, 'RoundnessMean') = {u};
    v = var(featureTable.Roundness(featureTable.Class == classArray(i)));
    statsTable(className, 'RoundnessVariance') = {v};
    
end
                 

tScoreArray = zeros([8 8 12]);
featureList = string(featureTable.Properties.VariableNames(3:14));
numberOfBean = 50;
for i = 1:numel(featureList)
    feature = char(featureList(i));
    for m = 1:numel(classArray)
        className1 = char(classArray(m));
        mean1 =  statsTable{className1, [feature, 'Mean']};
        varince1 = statsTable{className1, [feature, 'Variance']};
        for n = m+1:numel(classArray)
            className2 = char(classArray(n));
            mean2 =  statsTable{className2, [feature, 'Mean']};
            varince2 = statsTable{className2, [feature, 'Variance']};
            tScore = (mean1 - mean2) / sqrt((varince1/50) + (varince2/50));
            tScoreArray(m, n, i) = tScore; 
        end
    end
end

elapsedTime = toc;

function finalMask = morphological(imageMask)
    mark = imclose(imageMask, strel('disk', 5));
    mark = imdilate(mark, strel('disk', 2));
    mark = imfill(mark, 'holes');
    mark(:, 1:10) = 0; % trim left border
    finalMask = uint8(mark);
end

function score = evaluateMask(threshold, counts)
    range = 25; % range of calculation
    score = mean(counts(threshold-range : threshold+range));
end

        
