confThreshold = 0.90;
similarityThreshold = 0.45;
minFaceSizePx = 40;
blurVarThreshold = 60;
minBrightness = 30;
maxBrightness = 230;
maxMissedFrames = 10;
appearanceAlpha = 0.6;
iouMatchThreshold = 0.3;
useGPU = canUseGPU();

cams = webcamlist;
if isempty(cams)
    error('No cameras found. Please connect a camera.');
end
for i = 1:length(cams), fprintf('%d) %s\n', i, cams{i}); end
camIdx = input('Select camera number: ');
if isempty(camIdx) || camIdx < 1 || camIdx > numel(cams), error('Invalid camera index'); end
cam = webcam(camIdx);

availableRes = {};
try
    availableRes = cam.AvailableResolutions;
    if numel(availableRes) >= 3
        mediumIdx = round(numel(availableRes)/2);
    else
        mediumIdx = 1;
    end
    cam.Resolution = availableRes{mediumIdx};
catch
end

frame = snapshot(cam);
if size(frame,3) ~= 3 || all(frame(:)==0)
    clear cam;
    error('Selected camera produced black or empty image. Reconnect camera and try again.');
end

datasetFolder = fullfile(pwd, 'face_dataset');
if ~exist(datasetFolder,'dir'), mkdir(datasetFolder); end
modelFile = fullfile(pwd, 'highResFaceRecognitionModel.mat');
recogniseMode = false;
if exist(modelFile,'file')
    S = load(modelFile);
    if isfield(S,'newNet'), net = S.newNet;
    elseif isfield(S,'net'), net = S.net;
    else error('No network found in model file'); end
    if isfield(S,'inputSize'), inputSize = S.inputSize;
    else inputSize = net.Layers(1).InputSize(1:2); end
    personClasses = net.Layers(end).Classes;
    recogniseMode = true;
else
    net = resnet50;
    inputSize = net.Layers(1).InputSize(1:2);
end

captureMode = false;
resp = input(' Enter capture mode to build dataset? (y/n): ', 's');
if lower(resp) == 'y', captureMode = true; end

if captureMode
    personName = input('Enter person name (no spaces): ', 's');
    personFolder = fullfile(datasetFolder, personName);
    if ~exist(personFolder,'dir'), mkdir(personFolder); end
    numCapturesPerPose = input('How many images per pose (e.g., 5, 10, 20)? ');
    if isempty(numCapturesPerPose) || numCapturesPerPose <= 0, numCapturesPerPose = 5; end
    angles = {'front','left','right','up','down','45left','45right'};
    distances = [1,2,3];
    totalCaptures = numel(angles) * numel(distances) * numCapturesPerPose;
    disp(['Total images to capture: ', num2str(totalCaptures)]);
end

faceDetector = vision.CascadeObjectDetector('FrontalFaceCART', 'MergeThreshold', 6);
try
    camRes = regexp(cam.Resolution, '(\d+)x(\d+)', 'tokens', 'once');
    if ~isempty(camRes)
        w = str2double(camRes{1}); h = str2double(camRes{2});
        minSide = max(minFaceSizePx, round(min([w h]) * 0.06));
        faceDetector.MinSize = [minSide minSide];
    end
catch
end

pointTracker = vision.PointTracker('MaxBidirectionalError',2.0);
margin = 0.2;
frameCount = 0;
MAX_TRACK_FRAMES = 10;
capturedCount = 0;
showTracking = true;
autoCaptureInterval = 0.8;
lastCapTime = tic;
disp('Controls: T = toggle tracking, Q = quit');
hFig = figure('Name','HighRes Face Recognition','NumberTitle','off');
set(hFig,'CurrentCharacter',char(0));
keepRunning = true;

faceDetectorDL = [];
landmarkNet = [];
galleryEmbeddings = [];
galleryLabels = [];
if exist('faceDetectorDL.mat','file')
    S = load('faceDetectorDL.mat');
    if isfield(S,'detector'), faceDetectorDL = S.detector; end
end
if exist('landmarkNet.mat','file')
    S = load('landmarkNet.mat');
    if isfield(S,'landmarkNet'), landmarkNet = S.landmarkNet; end
end
if exist('gallery.mat','file')
    S = load('gallery.mat');
    if isfield(S,'galleryEmbeddings') && isfield(S,'galleryLabels')
        galleryEmbeddings = S.galleryEmbeddings;
        galleryLabels = S.galleryLabels;
    end
end

tracks = struct('id', {}, 'bbox', {}, 'age', {}, 'missed', {}, 'embedding', {}, 'label', {}, 'lastSim', {});
nextTrackID = 1;
angleIdx = 1; distIdx = 1; angleStr = ""; distVal = 1;

if captureMode
    captureMetaFolder = fullfile(personFolder, 'meta');
    if ~exist(captureMetaFolder,'dir'), mkdir(captureMetaFolder); end
end

while keepRunning && ishandle(hFig)
    frame = snapshot(cam);
    if size(frame,3) ~= 3 || all(frame(:)==0)
        imshow(zeros(256,256,3,'uint8'));
        title('Frame is black or unavailable! Check your camera/device.');
        drawnow;
        pause(0.5);
        continue;
    end
    gray = im2gray(frame);

    bboxes = [];
    scores = [];
    try
        if ~isempty(faceDetectorDL)
            try
                [bboxes, scores] = detect(faceDetectorDL, frame, 'Threshold', 0.3);
            catch
                bboxes = [];
                scores = [];
            end
        end
        if isempty(bboxes)
            bboxes = faceDetector(gray);
        end
        if isempty(bboxes)
            smaller = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',8);
            try smaller.MinSize = [round(minFaceSizePx*0.7) round(minFaceSizePx*0.7)]; end
            try tmp = smaller(gray); catch tmp = []; end
            if ~isempty(tmp), bboxes = tmp; end
        end
    catch
        try
            bboxes = faceDetector(gray);
        catch
            bboxes = [];
        end
    end

    if ~isempty(bboxes)
        sizes = bboxes(:,3:4);
        keepIdx = sizes(:,1) >= minFaceSizePx & sizes(:,2) >= minFaceSizePx;
        if any(~keepIdx)
            bboxes = bboxes(keepIdx,:);
        end
        if ~isempty(scores) && numel(scores)==size(bboxes,1)
            scores = scores(keepIdx);
        end
    end

    detections = [];
    if ~isempty(bboxes)
        for faceIdx = 1:size(bboxes,1)
            bbox = round(bboxes(faceIdx,:));
            faceRegion = safeExtract(frame, bbox, margin);
            faceNorm = imresize(faceRegion, inputSize);
            emb = [];
            try
                emb = computeEmbedding(net, faceNorm, 'avg_pool', useGPU);
            catch
                emb = [];
            end
            q = struct();
            q.isBlurred = isBlurred(faceNorm, blurVarThreshold);
            q.brightness = mean(im2gray(faceNorm),'all');
            q.tooBrightOrDark = (q.brightness < minBrightness) || (q.brightness > maxBrightness);
            q.landmarkCount = 0;
            detections(end+1).bbox = bbox;
            detections(end).crop = faceRegion;
            detections(end).aligned = faceNorm;
            detections(end).emb = emb;
            detections(end).quality = q;
            if ~isempty(scores) && length(scores) >= faceIdx
                detections(end).score = scores(faceIdx);
            else
                detections(end).score = [];
            end
        end
    end

    numTracks = numel(tracks);
    numDets = numel(detections);
    if numTracks == 0
        for d = 1:numDets
            tracks(end+1) = createTrack(detections(d), nextTrackID);
            nextTrackID = nextTrackID + 1;
        end
    elseif numDets == 0
        for t = 1:numTracks
            tracks(t).missed = tracks(t).missed + 1;
        end
    else
        costMat = zeros(numTracks, numDets) + inf;
        for t = 1:numTracks
            tbbox = tracks(t).bbox;
            for d = 1:numDets
                dbbox = detections(d).bbox;
                iou = bboxIoU(tbbox, dbbox);
                appSim = 0;
                if ~isempty(tracks(t).embedding) && ~isempty(detections(d).emb)
                    appSim = dot(tracks(t).embedding, detections(d).emb);
                    appSim = max(min(appSim,1), -1);
                end
                cost = (1 - iou) * 0.6 + (1 - (appSim+1)/2) * 0.4;
                if iou < 0.02
                    cost = cost + 1.0;
                end
                costMat(t,d) = cost;
            end
        end

        assignedTracks = false(1,numTracks);
        assignedDets = false(1,numDets);
        pairs = [];
        [sortedCosts, linearIdx] = sort(costMat(:),'ascend');
        for k = 1:numel(sortedCosts)
            if ~isfinite(sortedCosts(k)), break; end
            [t,d] = ind2sub(size(costMat), linearIdx(k));
            if assignedTracks(t) || assignedDets(d), continue; end
            iou = bboxIoU(tracks(t).bbox, detections(d).bbox);
            appSim = -1;
            if ~isempty(tracks(t).embedding) && ~isempty(detections(d).emb)
                appSim = dot(tracks(t).embedding, detections(d).emb);
            end
            if iou < iouMatchThreshold && appSim < (similarityThreshold - 0.15)
                continue;
            end
            pairs(end+1,:) = [t d];
            assignedTracks(t) = true; assignedDets(d) = true;
            if all(assignedTracks) && all(assignedDets), break; end
        end

        for p = 1:size(pairs,1)
            t = pairs(p,1); d = pairs(p,2);
            tracks(t).bbox = detections(d).bbox;
            tracks(t).age = tracks(t).age + 1;
            tracks(t).missed = 0;
            if ~isempty(detections(d).emb)
                if isempty(tracks(t).embedding)
                    tracks(t).embedding = detections(d).emb;
                else
                    tracks(t).embedding = (1-appearanceAlpha)*tracks(t).embedding + appearanceAlpha*detections(d).emb;
                    tracks(t).embedding = tracks(t).embedding / norm(tracks(t).embedding + 1e-10);
                end
            end
            if ~isempty(galleryEmbeddings) && ~isempty(tracks(t).embedding)
                [label, sim] = matchToGallery(galleryEmbeddings, galleryLabels, tracks(t).embedding, similarityThreshold);
                tracks(t).label = label;
                tracks(t).lastSim = sim;
            end
        end

        for d = 1:numDets
            if ~assignedDets(d)
                tracks(end+1) = createTrack(detections(d), nextTrackID);
                nextTrackID = nextTrackID + 1;
            end
        end

        for t = 1:numTracks
            if ~assignedTracks(t)
                tracks(t).missed = tracks(t).missed + 1;
            end
        end
    end

    aliveIdx = [];
    for t = 1:numel(tracks)
        if tracks(t).missed <= maxMissedFrames
            aliveIdx(end+1) = t;
        end
    end
    tracks = tracks(aliveIdx);

    if ~isempty(detections)
        for faceIdx = 1:numel(detections)
            bbox = detections(faceIdx).bbox;
            faceNorm = detections(faceIdx).aligned;
            if recogniseMode
                try
                    [predictedLabel, scores] = classify(net, faceNorm);
                    [maxScore, idx] = max(scores);
                    if maxScore < confThreshold
                        displayLabel = "Unknown";
                        frame = insertText(frame, bbox(1:2), "Unknown", 'FontSize',18,'BoxColor','red','TextColor','white');
                    else
                        displayLabel = string(personClasses(idx));
                        frame = insertText(frame, bbox(1:2), displayLabel, 'FontSize',18,'BoxColor','yellow','TextColor','black');
                    end
                    frame = insertShape(frame, 'Rectangle', bbox, 'Color', 'cyan', 'LineWidth', 2);
                catch
                end
            else
                frame = insertShape(frame, 'Rectangle', bbox, 'Color', 'cyan', 'LineWidth', 2);
            end

            if captureMode && capturedCount < totalCaptures
                if toc(lastCapTime) > autoCaptureInterval
                    try
                        poseNum = mod(capturedCount, numCapturesPerPose) + 1;
                        totalPoseIdx = floor(capturedCount / numCapturesPerPose);
                        angleIdx = mod(totalPoseIdx, numel(angles)) + 1;
                        distIdx = mod(floor(totalPoseIdx / numel(angles)), numel(distances)) + 1;
                        angleStr = angles{angleIdx};
                        distVal = distances(distIdx);
                        frame = insertText(frame, [15, 20], sprintf('Please look: %s angle, %dm from camera', angleStr, distVal), 'FontSize',20,'BoxColor','blue','TextColor','white');

                        landmarks = [];
                        if exist('landmarkNet','var') && ~isempty(landmarkNet)
                            try
                                pts = predictLandmarks(landmarkNet, detections(faceIdx).crop);
                                if ~isempty(pts)
                                    landmarks = pts;
                                end
                            catch
                                landmarks = [];
                            end
                        end
                        if isempty(landmarks)
                            try
                                gface = im2gray(detections(faceIdx).aligned);
                                corn = detectMinEigenFeatures(gface, 'NumStrongest', 68);
                                if ~isempty(corn)
                                    landmarks = corn.Location;
                                end
                            catch
                                landmarks = [];
                            end
                        end

                        kpoints = [];
                        try
                            gfaceFull = im2gray(detections(faceIdx).crop);
                            harris = detectHarrisFeatures(gfaceFull, 'MinQuality', 0.01);
                            if ~isempty(harris)
                                kpoints = harris.selectStrongest(20).Location;
                            end
                        catch
                            kpoints = [];
                        end

                        embToSave = detections(faceIdx).emb;
                        if isempty(embToSave)
                            try
                                embToSave = computeEmbedding(net, detections(faceIdx).aligned, 'avg_pool', useGPU);
                            catch
                                embToSave = [];
                            end
                        end

                        fnameBase = sprintf('%s_%s_%dm_%03d_face%02d', personName, angleStr, distVal, capturedCount+1, faceIdx);
                        fnameImg = fullfile(personFolder, [fnameBase, '.jpg']);
                        fnameMat = fullfile(captureMetaFolder, [fnameBase, '.mat']);
                        try
                            imwrite(im2uint8(detections(faceIdx).aligned), fnameImg);
                        catch
                            try imwrite(im2uint8(detections(faceIdx).crop), fnameImg); catch end
                        end

                        meta = struct();
                        meta.timestamp = datetime('now');
                        meta.person = personName;
                        meta.filename = fnameImg;
                        meta.bbox = detections(faceIdx).bbox;
                        meta.angle = angleStr;
                        meta.distance = distVal;
                        meta.captureIndex = capturedCount + 1;
                        meta.faceIdxInFrame = faceIdx;
                        meta.landmarks = landmarks;
                        meta.kernelPoints = kpoints;
                        meta.embedding = embToSave;
                        meta.quality = detections(faceIdx).quality;
                        meta.frameSize = size(frame);

                        try
                            save(fnameMat, 'meta', '-v7.3');
                        catch
                            try
                                mm = meta;
                                if isfield(mm,'embedding')
                                    mm.embedding = [];
                                end
                                save(fnameMat, 'mm');
                            catch
                            end
                        end

                        capturedCount = capturedCount + 1;
                        frame = insertText(frame, [15, 56], sprintf('Captured %d/%d',capturedCount,totalCaptures), 'FontSize',18,'BoxColor','black','TextColor','white');
                        lastCapTime = tic;

                        if ~isempty(embToSave)
                            embCol = embToSave(:);
                            if isempty(galleryEmbeddings)
                                galleryEmbeddings = embCol;
                            else
                                galleryEmbeddings(:,end+1) = embCol;
                            end
                            if isempty(galleryLabels)
                                galleryLabels = string(personName);
                            else
                                galleryLabels(end+1,1) = string(personName);
                            end
                        end

                        if capturedCount >= totalCaptures
                            disp('Auto-capture complete.');
                            captureMode = false;
                            try
                                save('gallery.mat', 'galleryEmbeddings', 'galleryLabels', '-v7.3');
                                disp('Gallery updated and saved.');
                            catch
                                warning('Failed saving gallery.mat');
                            end
                        end
                    catch MEcap
                        if exist('mm','var') && isfield(mm,'embedding'), mm = rmfield(mm,'embedding'); end
                    end
                end
            end
        end
    else
        imshow(zeros(256,256,3,'uint8'));
        title('No face detected.');
    end

    imshow(frame);
    drawnow;

    ch = get(hFig,'CurrentCharacter');
    if ~isempty(ch)
        switch lower(ch)
            case 'q'
                keepRunning = false;
            case 't'
                showTracking = ~showTracking;
        end
        set(hFig,'CurrentCharacter',char(0));
    end
    if ~ishandle(hFig), break; end
end

resp = input('Train model now? (y/n): ','s');
if lower(resp) == 'y'
    try
        trainHighResFaceRecognitionModel(datasetFolder);
    catch MEtrain
       warning(MEtrain.identifier, '%s', MEtrain.message);
    end
end

if exist('galleryEmbeddings','var') && ~isempty(galleryEmbeddings)
    try
        save('gallery.mat', 'galleryEmbeddings', 'galleryLabels', '-v7.3');
    catch
        warning('Could not save gallery.mat');
    end
end

clear cam
try release(pointTracker); catch end
close all
disp('Application ended.')

function faceRegion = safeExtract(frame,bbox,margin)
    x = max(1, round(bbox(1) - bbox(3)*margin));
    y = max(1, round(bbox(2) - bbox(4)*margin));
    w = min(size(frame,2)-x+1, round(bbox(3)*(1+2*margin)));
    h = min(size(frame,1)-y+1, round(bbox(4)*(1+2*margin)));
    if w <= 0 || h <= 0
        cx = round(size(frame,2)/2); cy = round(size(frame,1)/2);
        w = min(128, size(frame,2)); h = min(128, size(frame,1));
        x = max(1, cx - round(w/2)); y = max(1, cy - round(h/2));
    end
    region = frame(y:y+h-1, x:x+w-1, :);
    faceRegion = region;
end

function tr = createTrack(detection, id)
    tr.id = id;
    tr.bbox = detection.bbox;
    tr.age = 1;
    tr.missed = 0;
    if ~isempty(detection.emb), tr.embedding = detection.emb; else tr.embedding = []; end
    tr.label = "";
    tr.lastSim = 0;
end

function b = isBlurred(img, threshold)
    if nargin < 2, threshold = 60; end
    if size(img,3) == 3, g = rgb2gray(img); else g = img; end
    lap = imfilter(double(g), fspecial('laplacian',0.5), 'replicate');
    v = var(lap(:));
    b = (v < threshold);
end

function emb = computeEmbedding(net, img, layerName, useGPU)
    emb = [];
    if isempty(img)
        return;
    end
    if ~isa(img,'single'), imgIn = im2single(img); else imgIn = img; end
    imgIn = imresize(imgIn, net.Layers(1).InputSize(1:2));
    try
        if nargin>=4 && useGPU
            dl = gpuArray(imgIn);
            feat = activations(net, dl, layerName, 'OutputAs', 'columns');
            feat = gather(feat);
        else
            feat = activations(net, imgIn, layerName, 'OutputAs', 'columns');
        end
        feat = double(feat(:));
        if norm(feat) < 1e-10
            emb = [];
        else
            emb = feat / norm(feat + 1e-10);
        end
    catch
        try
            feat = activations(net, imgIn, layerName, 'OutputAs', 'columns');
            feat = double(feat(:));
            if norm(feat) < 1e-10
                emb = [];
            else
                emb = feat / norm(feat + 1e-10);
            end
        catch
            emb = [];
        end
    end
end

function iou = bboxIoU(b1, b2)
    xA = max(b1(1), b2(1));
    yA = max(b1(2), b2(2));
    xB = min(b1(1)+b1(3), b2(1)+b2(3));
    yB = min(b1(2)+b1(4), b2(2)+b2(4));
    interW = max(0, xB - xA + 1);
    interH = max(0, yB - yA + 1);
    inter = interW * interH;
    areaA = b1(3) * b1(4);
    areaB = b2(3) * b2(4);
    union = areaA + areaB - inter;
    if union <= 0, iou = 0; else iou = inter / union; end
end

function [label, sim] = matchToGallery(galleryEmb, galleryLabels, emb, th)
    label = "Unknown"; sim = 0;
    if isempty(galleryEmb) || isempty(emb), return; end
    try
        if size(galleryEmb,1) == numel(emb)
            sims = galleryEmb' * emb;
        else
            sims = (galleryEmb) * emb;
        end
        [sim, idx] = max(sims);
        if sim >= th
            label = galleryLabels(idx);
        else
            label = "Unknown";
        end
    catch
        label = "Unknown"; sim = 0;
    end
end

function pts4 = bbox2points(bbox)
    x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);
    pts4 = [x, y; x+w, y; x+w, y+h; x, y+h];
end

function pts = predictLandmarks(landmarkNet, crop)
    pts = [];
    try
        if isempty(landmarkNet), return; end
        inputSz = landmarkNet.InputSize(1:2);
        in = imresize(crop, inputSz);
        if ~isa(in,'single'), in = im2single(in); end
        out = predict(landmarkNet, in);
        if isvector(out) && mod(numel(out),2)==0
            pts = reshape(out, [2, numel(out)/2])';
        elseif ismatrix(out) && size(out,2)==2
            pts = out;
        end
    catch
        pts = [];
    end
end

function flag = canUseGPU()
    flag = false;
    try
        g = gpuDeviceCount;
        flag = (g>0);
    catch
        flag = false;
    end
end

function trainHighResFaceRecognitionModel(datasetPath)
    net = resnet50;
    inputSize = net.Layers(1).InputSize(1:2);
    imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    augs = imageDataAugmenter('RandXReflection', true, 'RandRotation', [-30 30], 'RandScale', [0.7 1.3], 'RandXTranslation', [-20 20], 'RandYTranslation', [-20 20]);
    augImds = augmentedImageDatastore(inputSize, imds, 'DataAugmentation', augs, 'ColorPreprocessing', 'none');
    [imdsTrain, imdsVal] = splitEachLabel(imds, 0.9, 'randomized');
    numClasses = numel(categories(imds.Labels));
    lgraph = layerGraph(net);
    newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'fcNew');
    newClassLayer = classificationLayer('Name', 'classoutput');
    lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);
    opts = trainingOptions('adam', 'MiniBatchSize', 16, 'MaxEpochs', 30, 'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', 'Plots', 'training-progress', 'ValidationData', augmentedImageDatastore(inputSize, imdsVal, 'DataAugmentation', augs, 'ColorPreprocessing', 'none'), 'ValidationFrequency', 25, 'Verbose', false);
    newNet = trainNetwork(augImds, lgraph, opts);
    save('highResFaceRecognitionModel.mat','newNet','inputSize');
    disp('High-resolution model trained and saved.');
    YPred = classify(newNet, augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'none'));
    confusionchart(imdsVal.Labels, YPred);
end
