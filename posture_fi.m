clear; clc; close all;

confThreshold = 0.90;
MAX_PEOPLE = 5;
LEARNING_MODE = true;
SECURITY_MODE = true;

cams = webcamlist;
if isempty(cams)
    error('No cameras found. Please connect a camera.');
end
fprintf('Available cameras:\n');
for i = 1:length(cams)
    fprintf('%d) %s\n', i, cams{i});
end
camIdx = input('Select camera number: ');
if camIdx < 1 || camIdx > numel(cams)
    error('Invalid camera selection.');
end

cam = webcam(camIdx);
availableRes = cam.AvailableResolutions;
if numel(availableRes) >= 3
    resIdx = round(numel(availableRes)/2);
else
    resIdx = 1;
end
cam.Resolution = availableRes{resIdx};
fprintf('Camera resolution: %s\n', cam.Resolution);

frame = snapshot(cam);
if size(frame,3) ~= 3 || all(frame(:)==0)
    error('Camera validation failed');
end

datasetFolder = fullfile(pwd, 'face_dataset');
if ~exist(datasetFolder, 'dir'), mkdir(datasetFolder); end
modelFile = fullfile(pwd, 'highResFaceRecognitionModel.mat');

if exist(modelFile, 'file')
    S = load(modelFile);
    if isfield(S, 'newNet')
        net = S.newNet;
    elseif isfield(S, 'net')
        net = S.net;
    else 
        error('No network found in model file');
    end
    if isfield(S,'inputSize')
        inputSize = S.inputSize;
    else 
        inputSize = net.Layers(1).InputSize(1:2);
    end
    personClasses = net.Layers(end).Classes;
    recogniseMode = true;
    fprintf('Face recognition: %d classes loaded\n', numel(personClasses));
else
    recogniseMode = false;
    fprintf('No face recognition model found\n');
end

poseEnabled = false;
poseDetector = [];
peopleDetect = [];
peopleDetectEnabled = false;

try
    poseDetector = hrnetObjectKeypointDetector;
    poseDetector.Threshold = 0.15;
    poseEnabled = true;
    fprintf('Advanced pose detection ready\n');

    try
        peopleDetect = peopleDetectorACF;
        peopleDetectEnabled = true;
        fprintf('People detector ready\n');
    catch
        fprintf('Using face-based people detection\n');
        peopleDetectEnabled = false;
    end
catch ME
    poseEnabled = false;
    fprintf('Pose detection unavailable: %s\n', ME.message);
end

COMPREHENSIVE_ACTIVITIES = {
    'Standing'; 'Sitting'; 'Walking'; 'Running'; 'Jogging'; 'Jumping'; 'Crouching';
    'Drinking_Water'; 'Drinking_Coffee'; 'Eating'; 'Phone_Call'; 'Texting';
    'Writing'; 'Typing'; 'Reading'; 'Studying'; 'Working_Computer';
    'Waving'; 'Pointing'; 'Clapping'; 'Thumbs_Up'; 'Victory_Pose'; 'Hands_Up';
    'Fighting_Stance'; 'Defensive_Pose'; 'Crossed_Arms'; 'Arms_Folded';
    'Stretching'; 'Exercise_Pose'; 'Yoga_Pose'; 'Meditation'; 'Relaxing';
    'Thinking_Pose'; 'Head_Scratch'; 'Face_Touch'; 'Chin_Rest';
    'Greeting'; 'Handshake'; 'Hugging'; 'High_Five'; 'Conversation';
    'Presentation'; 'Teaching'; 'Explaining'; 'Demonstrating';
    'Suspicious_Activity'; 'Loitering'; 'Looking_Around'; 'Hiding';
    'Carrying_Object'; 'Lifting'; 'Pulling'; 'Pushing'; 'Reaching';
    'Bending'; 'Leaning'; 'Slouching'; 'Good_Posture'; 'Poor_Posture';
    'Fidgeting'; 'Restless'; 'Calm'; 'Focused'; 'Distracted';
    'Custom_Activity_1'; 'Custom_Activity_2'; 'Custom_Activity_3'
};

SECURITY_ALERTS = {
    'Normal_Behavior'; 'Suspicious_Loitering'; 'Aggressive_Posture';
    'Concealed_Object'; 'Unusual_Movement'; 'Restricted_Area_Entry';
    'Group_Formation'; 'Extended_Stay'; 'Erratic_Behavior'
};

poseHistory = containers.Map();
individualLogs = containers.Map();
activityLearning = containers.Map();
securityEvents = [];
MAX_POSE_HISTORY = 20;
CSV_LOG_FILE = sprintf('posture_log_%s.csv', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));

initializeCSVLog(CSV_LOG_FILE);

captureMode = false;
resp = input('Enter face capture mode? (y/n): ', 's');
if strcmpi(resp, 'y')
    captureMode = true;
    personName = input('Person name: ', 's');
    personFolder = fullfile(datasetFolder, personName);
    if ~exist(personFolder, 'dir'), mkdir(personFolder); end
    numCapturesPerPose = input('Images per pose (default 15): ');
    if isempty(numCapturesPerPose) || numCapturesPerPose <= 0
        numCapturesPerPose = 15;
    end
    angles = {'front', 'left', 'right', 'up', 'down', '45left', '45right', 'profile_left', 'profile_right'};
    distances = [1, 1.5, 2, 2.5, 3];
    lighting = {'normal', 'bright', 'dim'};
    totalCaptures = numel(angles) * numel(distances) * numel(lighting) * numCapturesPerPose;
    fprintf('Total captures required: %d\n', totalCaptures);
end

faceDetector = vision.CascadeObjectDetector();
pointTracker = vision.PointTracker('MaxBidirectionalError', 2.0);
margin = 0.15;

if ~exist('net', 'var')
    net = resnet50;
    inputSize = net.Layers(1).InputSize(1:2);
end

frameCount = 0;
capturedCount = 0;
autoCaptureInterval = 1.0;
lastCapTime = tic;
sessionStartTime = now;

figWidth = 1600; figHeight = 1000; videoWidth = 1000; panelWidth = figWidth - videoWidth;

hMainFig = figure('Name', 'Comprehensive Face + Posture Analysis System', ...
                  'Position', [50, 50, figWidth, figHeight], ...
                  'NumberTitle', 'off', 'Resize', 'off', ...
                  'CloseRequestFcn', @mainCloseCallback);

hVideoAxes = axes('Parent', hMainFig, 'Units', 'pixels', ...
                  'Position', [20, 20, videoWidth-40, figHeight-100]);

hControlPanel = uipanel('Parent', hMainFig, 'Units', 'pixels', ...
                        'Position', [videoWidth, 20, panelWidth-20, figHeight-40], ...
                        'Title', 'System Control & Analytics', 'FontSize', 12, ...
                        'BackgroundColor', [0.95 0.95 0.95]);

hPeoplePanel = uipanel('Parent', hControlPanel, 'Units', 'pixels', ...
                       'Position', [10, 700, panelWidth-50, 250], ...
                       'Title', 'Detected People (Live)', 'FontSize', 11, ...
                       'BackgroundColor', 'white');

hActivityPanel = uipanel('Parent', hControlPanel, 'Units', 'pixels', ...
                         'Position', [10, 400, panelWidth-50, 290], ...
                         'Title', 'Activity Analysis', 'FontSize', 11, ...
                         'BackgroundColor', 'white');

hSecurityPanel = uipanel('Parent', hControlPanel, 'Units', 'pixels', ...
                         'Position', [10, 200, panelWidth-50, 190], ...
                         'Title', 'Security Monitor', 'FontSize', 11, ...
                         'BackgroundColor', [1.0 0.95 0.95]);

hStatsPanel = uipanel('Parent', hControlPanel, 'Units', 'pixels', ...
                      'Position', [10, 10, panelWidth-50, 180], ...
                      'Title', 'System Statistics', 'FontSize', 11, ...
                      'BackgroundColor', [0.95 1.0 0.95]);

hPeopleList = uicontrol('Parent', hPeoplePanel, 'Style', 'listbox', ...
                        'Units', 'pixels', 'Position', [10, 40, panelWidth-80, 180], ...
                        'FontSize', 9, 'FontName', 'Courier New', ...
                        'BackgroundColor', 'white');

hConfidenceText = uicontrol('Parent', hPeoplePanel, 'Style', 'text', ...
                            'Units', 'pixels', 'Position', [10, 10, panelWidth-80, 25], ...
                            'String', 'Confidence Levels', 'FontSize', 9, ...
                            'BackgroundColor', 'white', 'HorizontalAlignment', 'left');

hActivityList = uicontrol('Parent', hActivityPanel, 'Style', 'listbox', ...
                          'Units', 'pixels', 'Position', [10, 40, panelWidth-80, 220], ...
                          'FontSize', 9, 'FontName', 'Courier New', ...
                          'BackgroundColor', 'white');

hActivityStats = uicontrol('Parent', hActivityPanel, 'Style', 'text', ...
                           'Units', 'pixels', 'Position', [10, 10, panelWidth-80, 25], ...
                           'String', 'Activity Duration Tracking', 'FontSize', 9, ...
                           'BackgroundColor', 'white', 'HorizontalAlignment', 'left');

hSecurityList = uicontrol('Parent', hSecurityPanel, 'Style', 'listbox', ...
                          'Units', 'pixels', 'Position', [10, 40, panelWidth-80, 120], ...
                          'FontSize', 9, 'FontName', 'Courier New', ...
                          'BackgroundColor', [1.0 0.98 0.98]);

hSecurityLevel = uicontrol('Parent', hSecurityPanel, 'Style', 'text', ...
                           'Units', 'pixels', 'Position', [10, 10, panelWidth-80, 25], ...
                           'String', 'Security Level: Normal', 'FontSize', 10, ...
                           'BackgroundColor', [1.0 0.95 0.95], 'HorizontalAlignment', 'left');

hSystemStats = uicontrol('Parent', hStatsPanel, 'Style', 'text', ...
                         'Units', 'pixels', 'Position', [10, 10, panelWidth-80, 140], ...
                         'FontSize', 9, 'FontName', 'Courier New', ...
                         'BackgroundColor', [0.95 1.0 0.95], 'HorizontalAlignment', 'left');

hLearningButton = uicontrol('Parent', hMainFig, 'Style', 'pushbutton', ...
                            'Units', 'pixels', 'Position', [videoWidth+50, figHeight-30, 120, 25], ...
                            'String', 'Learn Activity', 'FontSize', 10, ...
                            'Callback', @learnActivityCallback);

hExportButton = uicontrol('Parent', hMainFig, 'Style', 'pushbutton', ...
                          'Units', 'pixels', 'Position', [videoWidth+180, figHeight-30, 120, 25], ...
                          'String', 'Export CSV', 'FontSize', 10, ...
                          'Callback', @exportCSVCallback);

trackedPeople = containers.Map();
nextPersonID = 1;
set(hMainFig, 'CurrentCharacter', char(0));
keepRunning = true;
showPoseOverlay = true;
showActivityLabels = true;
show3DVisualization = true;

avgProcessTime = 0.1;
totalPeopleDetected = 0;
totalActivitiesLogged = 0;
securityAlertLevel = 'Normal';

fprintf('System initialized. Starting comprehensive analysis...\n');
fprintf('Controls: Q=Quit, P=Toggle Pose, A=Toggle Activities, V=Toggle 3D, L=Learn Mode\n');

angleIdx = 1; distIdx = 1; lightIdx = 1;

while keepRunning && ishandle(hMainFig)
    frameCount = frameCount + 1;
    loopStart = tic;

    try
        frame = snapshot(cam);
        if size(frame,3) ~= 3 || all(frame(:)==0)
            continue;
        end

        originalFrame = frame;
        gray = im2gray(frame);

        bboxes = faceDetector(gray);
        currentPeople = {};
        currentActivities = {};
        currentSecurity = {};
        faceResults = [];

        if ~isempty(bboxes)
            for faceIdx = 1:size(bboxes, 1)
                bbox = bboxes(faceIdx, :);

                try
                    faceRegion = extractFaceRegion(frame, bbox, margin, inputSize);
                    faceNorm = imresize(faceRegion, inputSize);

                    if recogniseMode
                        [predictedLabel, scores] = classify(net, faceNorm);
                        [maxScore, idx] = max(scores);

                        if maxScore < confThreshold
                            displayLabel = "Unknown";
                            boxColor = 'red';
                            securityRisk = 'Medium';
                        else
                            displayLabel = string(personClasses(idx));
                            boxColor = 'yellow';
                            securityRisk = 'Low';
                        end

                        frame = insertText(frame, bbox(1:2), char(displayLabel), ...
                            'FontSize', 18, 'BoxColor', boxColor, 'TextColor', 'white');
                        frame = insertShape(frame, 'Rectangle', bbox, ...
                            'Color', 'cyan', 'LineWidth', 3);

                        personID = assignPersonID(bbox, trackedPeople, nextPersonID);
                        if personID >= nextPersonID
                            nextPersonID = personID + 1;
                            totalPeopleDetected = totalPeopleDetected + 1;
                        end

                        faceResults = [faceResults; struct('bbox', bbox, 'label', displayLabel, ...
                                                          'confidence', maxScore, 'personID', personID, ...
                                                          'securityRisk', securityRisk)];

                        currentPeople{end+1} = sprintf('ID%d: %s (%.1f%%) [%s Risk]', ...
                                                       personID, char(displayLabel), maxScore*100, securityRisk);

                        updateIndividualLog(individualLogs, personID, displayLabel, 'Face_Detected', maxScore, frameCount);
                    end

                    if captureMode && capturedCount < totalCaptures
                        if toc(lastCapTime) > autoCaptureInterval
                            poseInCycle = mod(capturedCount, numCapturesPerPose) + 1;
                            totalIdx = floor(capturedCount / numCapturesPerPose);

                            angleIdx = mod(totalIdx, numel(angles)) + 1;
                            distIdx = mod(floor(totalIdx / numel(angles)), numel(distances)) + 1;
                            lightIdx = mod(floor(totalIdx / (numel(angles) * numel(distances))), numel(lighting)) + 1;

                            angleStr = angles{angleIdx};
                            distVal = distances(distIdx);
                            lightStr = lighting{lightIdx};

                            frame = insertText(frame, [15, 20], ...
                                sprintf('Pose: %s | Dist: %.1fm | Light: %s | %d/%d', ...
                                        angleStr, distVal, lightStr, poseInCycle, numCapturesPerPose), ...
                                'FontSize', 16, 'BoxColor', 'blue', 'TextColor', 'white');

                            fname = sprintf('%s_%s_%.1fm_%s_%03d.jpg', personName, angleStr, distVal, lightStr, capturedCount+1);
                            imwrite(faceNorm, fullfile(personFolder, fname));
                            capturedCount = capturedCount + 1;

                            frame = insertText(frame, [15, 50], ...
                                sprintf('Captured: %d/%d (%.1f%% complete)', capturedCount, totalCaptures, ...
                                        (capturedCount/totalCaptures)*100), ...
                                'FontSize', 14, 'BoxColor', 'black', 'TextColor', 'white');

                            lastCapTime = tic;

                            if capturedCount >= totalCaptures
                                fprintf('Enhanced capture sequence completed: %d images\n', capturedCount);
                                captureMode = false;
                            end
                        end
                    end
                catch ME
                    fprintf('Face processing error: %s\n', ME.message);
                end
            end

            if poseEnabled && showPoseOverlay
                try
                    humanBboxes = [];

                    if peopleDetectEnabled
                        try
                            [humanBboxes, ~, ~] = detect(peopleDetect, originalFrame, 'Threshold', 0.25);
                        catch
                        end
                    end

                    if isempty(humanBboxes) && ~isempty(bboxes)
                        humanBboxes = expandFaceBboxForBody(bboxes, size(originalFrame));
                    end

                    if ~isempty(humanBboxes)
                        [keypoints, keypointScores, validity] = detect(poseDetector, originalFrame, humanBboxes);

                        for personIdx = 1:min(size(humanBboxes, 1), MAX_PEOPLE)
                            if personIdx <= size(keypoints, 3)
                                currentKeypoints = keypoints(:,:,personIdx);
                                currentValidity = validity(:,personIdx);
                                currentScores = keypointScores(:,personIdx);

                                if sum(currentValidity) >= 12
                                    if show3DVisualization
                                        frame = insert3DPoseVisualization(frame, currentKeypoints, currentValidity, currentScores);
                                    else
                                        frame = insertObjectKeypoints(frame, currentKeypoints, ...
                                            'KeypointVisibility', currentValidity, ...
                                            'Connections', poseDetector.KeypointConnections, ...
                                            'ConnectionColor', 'green', 'KeypointColor', 'yellow', ...
                                            'KeypointSize', 5, 'LineWidth', 3);
                                    end
                                end

                                if showActivityLabels && sum(currentValidity) >= 10
                                    [activity, confidence, biomechanics] = comprehensiveActivityRecognition(...
                                        currentKeypoints, currentValidity, personIdx, poseHistory, frameCount, activityLearning);

                                    securityAssessment = assessSecurityRisk(activity, confidence, biomechanics, frameCount);

                                    bboxH = humanBboxes(personIdx, :);
                                    activityColor = getAdvancedActivityColor(activity);

                                    frame = insertText(frame, [bboxH(1), max(1, bboxH(2)-45)], ...
                                        sprintf('%s (%.0f%%)', activity, confidence*100), ...
                                        'FontSize', 12, 'BoxColor', activityColor, 'TextColor', 'white');

                                    associatedFace = findClosestFace(bboxH, faceResults);
                                    if ~isempty(associatedFace)
                                        personName = char(associatedFace.label);
                                        personID = associatedFace.personID;

                                        frame = insertText(frame, [bboxH(1), max(1, bboxH(2)-65)], ...
                                            sprintf('ID%d: %s', personID, personName), ...
                                            'FontSize', 11, 'BoxColor', 'green', 'TextColor', 'white');

                                        currentActivities{end+1} = sprintf('ID%d %s: %s (%.0f%%) [%s]', ...
                                            personID, personName, activity, confidence*100, biomechanics.posture);

                                        updateIndividualLog(individualLogs, personID, personName, activity, confidence, frameCount);
                                        logToCSV(CSV_LOG_FILE, frameCount, personID, personName, activity, confidence, biomechanics);
                                        totalActivitiesLogged = totalActivitiesLogged + 1;

                                        if strcmp(securityAssessment.level, 'High') || strcmp(securityAssessment.level, 'Critical')
                                            securityEvents = [securityEvents; struct('timestamp', now, 'personID', personID, ...
                                                                                    'activity', activity, 'risk', securityAssessment.level, ...
                                                                                    'details', securityAssessment.details)];
                                            securityAlertLevel = securityAssessment.level;

                                            frame = insertText(frame, [bboxH(1), max(1, bboxH(2)-85)], ...
                                                sprintf('SECURITY: %s', securityAssessment.level), ...
                                                'FontSize', 10, 'BoxColor', 'red', 'TextColor', 'white');
                                        end

                                        currentSecurity{end+1} = sprintf('ID%d: %s Risk - %s', ...
                                            personID, securityAssessment.level, securityAssessment.details);
                                    else
                                        currentActivities{end+1} = sprintf('Person%d: %s (%.0f%%) [%s]', ...
                                            personIdx, activity, confidence*100, biomechanics.posture);

                                        updateIndividualLog(individualLogs, personIdx, 'Unknown', activity, confidence, frameCount);
                                        logToCSV(CSV_LOG_FILE, frameCount, personIdx, 'Unknown', activity, confidence, biomechanics);
                                    end
                                end
                            end
                        end
                    end
                catch ME
                    if frameCount <= 5
                        fprintf('Pose processing error: %s\n', ME.message);
                    end
                end
            end
        end

        imshow(frame, 'Parent', hVideoAxes);
        title(hVideoAxes, sprintf('Comprehensive Analysis | Frame: %d | FPS: %.1f | People: %d', ...
              frameCount, 1/max(avgProcessTime,0.01), length(faceResults)), 'FontSize', 12);

        updateUserInterface(hPeopleList, hActivityList, hSecurityList, hSystemStats, hSecurityLevel, ...
                           currentPeople, currentActivities, currentSecurity, frameCount, ...
                           totalPeopleDetected, totalActivitiesLogged, securityAlertLevel, avgProcessTime);

        drawnow;

        ch = get(hMainFig, 'CurrentCharacter');
        if ~isempty(ch)
            switch lower(ch)
                case 'q'
                    keepRunning = false;
                case 'p'
                    if poseEnabled
                        showPoseOverlay = ~showPoseOverlay;
                        fprintf('Pose overlay: %s\n', iff(showPoseOverlay, 'ON', 'OFF'));
                    end
                case 'a'
                    if poseEnabled
                        showActivityLabels = ~showActivityLabels;
                        fprintf('Activity labels: %s\n', iff(showActivityLabels, 'ON', 'OFF'));
                    end
                case 'v'
                    if poseEnabled
                        show3DVisualization = ~show3DVisualization;
                        fprintf('3D visualization: %s\n', iff(show3DVisualization, 'ON', 'OFF'));
                    end
                case 'l'
                    LEARNING_MODE = ~LEARNING_MODE;
                    fprintf('Learning mode: %s\n', iff(LEARNING_MODE, 'ON', 'OFF'));
            end
            set(hMainFig, 'CurrentCharacter', char(0));
        end

        if ~ishandle(hMainFig)
            break;
        end

    catch ME
        fprintf('Main loop error at frame %d: %s\n', frameCount, ME.message);
        if contains(ME.message, 'Interrupted')
            break;
        end
    end

    frameTime = toc(loopStart);
    avgProcessTime = 0.85 * avgProcessTime + 0.15 * frameTime;
end

if captureMode && capturedCount > 0
    resp = input('Train enhanced face recognition model? (y/n): ', 's');
    if strcmpi(resp, 'y')
        trainEnhancedFaceRecognitionModel(datasetFolder);
    end
end

exportFinalReport(individualLogs, securityEvents, sessionStartTime, frameCount, CSV_LOG_FILE);

try
    if exist('hMainFig','var') && ishghandle(hMainFig)
        set(hMainFig, 'CloseRequestFcn', '');
        delete(hMainFig);
    end
catch
end

clear cam
release(pointTracker)
close all force

fprintf('\n=== SESSION SUMMARY ===\n');
fprintf('Duration: %.1f minutes\n', (now - sessionStartTime) * 24 * 60);
fprintf('Frames processed: %d\n', frameCount);
fprintf('People detected: %d\n', totalPeopleDetected);
fprintf('Activities logged: %d\n', totalActivitiesLogged);
fprintf('Security events: %d\n', length(securityEvents));
fprintf('Data exported to: %s\n', CSV_LOG_FILE);
fprintf('========================\n');

disp('Comprehensive Face Recognition + Posture Analysis System ended.');

function faceRegion = extractFaceRegion(frame, bbox, margin, outputSize)
    try
        [frameH, frameW, ~] = size(frame);
        x = max(1, round(bbox(1) - bbox(3) * margin));
        y = max(1, round(bbox(2) - bbox(4) * margin));
        w = min(frameW - x + 1, round(bbox(3) * (1 + 2 * margin)));
        h = min(frameH - y + 1, round(bbox(4) * (1 + 2 * margin)));
        region = frame(y:y+h-1, x:x+w-1, :);
        faceRegion = imresize(region, outputSize);
    catch
        x = max(1, round(bbox(1))); y = max(1, round(bbox(2)));
        w = min(size(frame,2) - x + 1, round(bbox(3)));
        h = min(size(frame,1) - y + 1, round(bbox(4)));
        region = frame(y:y+h-1, x:x+w-1, :);
        faceRegion = imresize(region, outputSize);
    end
end

function trainEnhancedFaceRecognitionModel(datasetPath)
    fprintf('Training enhanced face recognition model...\n');
    try
        net = resnet50;
        inputSize = net.Layers(1).InputSize(1:2);
        imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

        fprintf('Dataset: %d images across %d classes\n', numel(imds.Files), numel(categories(imds.Labels)));

        augs = imageDataAugmenter('RandXReflection', true, 'RandRotation', [-25 25], ...
            'RandScale', [0.75 1.25], 'RandXTranslation', [-15 15], 'RandYTranslation', [-15 15], ...
            'RandXShear', [-5 5], 'RandYShear', [-5 5]);

        augImds = augmentedImageDatastore(inputSize, imds, 'DataAugmentation', augs, 'ColorPreprocessing', 'none');
        [imdsTrain, imdsVal] = splitEachLabel(imds, 0.85, 'randomized');

        numClasses = numel(categories(imds.Labels));
        lgraph = layerGraph(net);
        newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'fcNew', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
        newClassLayer = classificationLayer('Name', 'classoutput');
        lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
        lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);

        opts = trainingOptions('adam', 'MiniBatchSize', 16, 'MaxEpochs', 35, 'InitialLearnRate', 1e-4, ...
            'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 12, ...
            'Shuffle', 'every-epoch', 'Plots', 'training-progress', ...
            'ValidationData', augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'none'), ...
            'ValidationFrequency', 30, 'Verbose', true);

        newNet = trainNetwork(augImds, lgraph, opts);
        save('highResFaceRecognitionModel.mat', 'newNet', 'inputSize', '-v7.3');

        YPred = classify(newNet, augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'none'));
        accuracy = sum(YPred == imdsVal.Labels) / numel(imdsVal.Labels);
        fprintf('Enhanced model trained | Accuracy: %.2f%%\n', accuracy * 100);

        if numel(categories(imdsVal.Labels)) <= 10
            figure; confusionchart(imdsVal.Labels, YPred); title('Enhanced Model Performance');
        end

    catch ME
        fprintf('Training failed: %s\n', ME.message);
    end
end

function pts4 = bbox2points(bbox)
    x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);
    pts4 = [x, y; x+w, y; x+w, y+h; x, y+h];
end

function expandedBboxes = expandFaceBboxForBody(faceBboxes, frameSize)
    expandedBboxes = [];
    if isempty(faceBboxes), return; end

    frameH = frameSize(1); frameW = frameSize(2);
    expandedBboxes = zeros(size(faceBboxes));

    for i = 1:size(faceBboxes, 1)
        bbox = faceBboxes(i, :);
        bodyHeight = min(bbox(4) * 7.5, frameH);
        bodyWidth = min(bbox(3) * 3.2, frameW);

        newX = max(1, bbox(1) - (bodyWidth - bbox(3)) / 2);
        newY = max(1, bbox(2) - bbox(4) * 0.15);

        newX = max(1, min(newX, frameW - bodyWidth + 1));
        newY = max(1, min(newY, frameH - bodyHeight + 1));
        bodyWidth = min(bodyWidth, frameW - newX + 1);
        bodyHeight = min(bodyHeight, frameH - newY + 1);

        expandedBboxes(i, :) = [newX, newY, bodyWidth, bodyHeight];
    end
end

function personID = assignPersonID(bbox, trackedPeople, nextID)
    bboxCenter = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
    minDistance = inf; assignedID = nextID;

    keysList = trackedPeople.keys;
    for i = 1:length(keysList)
        person = trackedPeople(keysList{i});
        if (now - person.lastSeen) * 24 * 3600 < 30
            distance = norm(bboxCenter - person.lastCenter);
            if distance < 80 && distance < minDistance
                minDistance = distance; assignedID = person.id;
            end
        end
    end

    personKey = sprintf('person_%d', assignedID);
    trackedPeople(personKey) = struct('id', assignedID, 'lastCenter', bboxCenter, 'lastSeen', now);
    personID = assignedID;
end

function [activity, confidence, biomechanics] = comprehensiveActivityRecognition(keypoints, validity, personId, poseHistory, frameCount, activityLearning)
    activity = 'Standing'; confidence = 0.6;
    biomechanics = struct('posture', 'Neutral', 'balance', 0.5, 'energy', 0.5, 'coordination', 0.5);

    try
        if sum(validity) < 10
            activity = 'Unknown'; confidence = 0.0; return;
        end

        biomechanics = calculateBiomechanics(keypoints, validity);

        historyKey = sprintf('person_%d', personId);
        currentPose = struct('keypoints', keypoints, 'validity', validity, 'frame', frameCount, 'biomechanics', biomechanics);

        if isKey(poseHistory, historyKey)
            history = poseHistory(historyKey);
            history(end+1) = currentPose;
            if length(history) > MAX_POSE_HISTORY
                history = history(end-MAX_POSE_HISTORY+1:end);
            end
            poseHistory(historyKey) = history;
        else
            poseHistory(historyKey) = currentPose; return;
        end

        history = poseHistory(historyKey);
        if length(history) < 3, return; end

        [activity, confidence] = detectComprehensiveActivity(keypoints, validity, history, biomechanics, activityLearning);

    catch ME
        activity = 'Error'; confidence = 0.0;
        fprintf('Activity recognition error: %s\n', ME.message);
    end
end

function biomechanics = calculateBiomechanics(keypoints, validity)
    biomechanics = struct('posture', 'Neutral', 'balance', 0.5, 'energy', 0.5, 'coordination', 0.5);

    try
        if sum(validity) < 10, return; end

        if validity(6) && validity(7) && validity(12) && validity(13)
            shoulderLine = keypoints(6, :) - keypoints(7, :);
            hipLine = keypoints(12, :) - keypoints(13, :);
            shoulderAngle = atan2d(shoulderLine(2), shoulderLine(1));
            hipAngle = atan2d(hipLine(2), hipLine(1));

            postureDiff = abs(shoulderAngle - hipAngle);
            if postureDiff < 5
                biomechanics.posture = 'Excellent';
                biomechanics.balance = 0.9;
            elseif postureDiff < 15
                biomechanics.posture = 'Good';
                biomechanics.balance = 0.7;
            elseif postureDiff < 30
                biomechanics.posture = 'Fair';
                biomechanics.balance = 0.5;
            else
                biomechanics.posture = 'Poor';
                biomechanics.balance = 0.3;
            end
        end

        if validity(1) && validity(6) && validity(7)
            neckPos = keypoints(1, :);
            shoulderCenter = mean([keypoints(6, :); keypoints(7, :)], 1);
            neckAlignment = norm(neckPos - shoulderCenter);
            biomechanics.energy = max(0.1, min(1.0, 1.0 - neckAlignment/100));
        end

        validJoints = sum(validity);
        biomechanics.coordination = validJoints / 17;

    catch
        biomechanics.posture = 'Unknown';
    end
end

function [activity, confidence] = detectComprehensiveActivity(keypoints, validity, history, biomechanics, activityLearning)
    activity = 'Standing'; confidence = 0.6;

    try
        if isDrinkingPose(keypoints, validity), activity = 'Drinking_Water'; confidence = 0.88; return; end
        if isPhoneCallPose(keypoints, validity), activity = 'Phone_Call'; confidence = 0.85; return; end
        if isTypingPose(keypoints, validity), activity = 'Typing'; confidence = 0.82; return; end
        if isFightingStance(keypoints, validity), activity = 'Fighting_Stance'; confidence = 0.92; return; end
        if isWritingPose(keypoints, validity), activity = 'Writing'; confidence = 0.78; return; end
        if isHandsUpPose(keypoints, validity), activity = 'Hands_Up'; confidence = 0.88; return; end
        if isVictoryPose(keypoints, validity), activity = 'Victory_Pose'; confidence = 0.82; return; end
        if isCrossedArmsPose(keypoints, validity), activity = 'Crossed_Arms'; confidence = 0.75; return; end
        if isPointingPose(keypoints, validity), activity = 'Pointing'; confidence = 0.72; return; end
        if isClappingPose(keypoints, validity), activity = 'Clapping'; confidence = 0.88; return; end
        if isReadingPose(keypoints, validity), activity = 'Reading'; confidence = 0.75; return; end
        if isThinkingPose(keypoints, validity), activity = 'Thinking_Pose'; confidence = 0.70; return; end
        if isSuspiciousActivity(keypoints, validity, history), activity = 'Suspicious_Activity'; confidence = 0.85; return; end
        if isCarryingObject(keypoints, validity), activity = 'Carrying_Object'; confidence = 0.80; return; end
        if isStretchingPose(keypoints, validity), activity = 'Stretching'; confidence = 0.77; return; end

        [velocity, verticalVar, armMovement, spatialMovement] = extractAdvancedMovementFeatures(history);

        if velocity > 15, activity = 'Running'; confidence = 0.85;
        elseif velocity > 8, activity = 'Jogging'; confidence = 0.80;
        elseif velocity > 3, activity = 'Walking'; confidence = 0.78;
        elseif verticalVar > 15, activity = 'Jumping'; confidence = 0.88;
        elseif armMovement > 25, activity = 'Waving'; confidence = 0.72;
        elseif spatialMovement > 20, activity = 'Fidgeting'; confidence = 0.68;
        elseif isSittingPose(keypoints, validity), activity = 'Sitting'; confidence = 0.82;
        elseif isCrouchingPose(keypoints, validity), activity = 'Crouching'; confidence = 0.75;
        else
            if strcmp(biomechanics.posture, 'Excellent') || strcmp(biomechanics.posture, 'Good')
                activity = 'Good_Posture'; confidence = 0.70;
            elseif strcmp(biomechanics.posture, 'Poor')
                activity = 'Poor_Posture'; confidence = 0.75;
            else
                activity = 'Standing'; confidence = 0.65;
            end
        end

    catch
        activity = 'Unknown'; confidence = 0.0;
    end
end

function isDrinking = isDrinkingPose(keypoints, validity)
    isDrinking = false;
    if validity(1) && validity(10) && validity(11)
        noseY = keypoints(1, 2); leftWristY = keypoints(10, 2); rightWristY = keypoints(11, 2);
        if abs(leftWristY - noseY) < 40 || abs(rightWristY - noseY) < 40, isDrinking = true; end
    end
end

function isPhone = isPhoneCallPose(keypoints, validity)
    isPhone = false;
    if validity(2) && validity(3) && validity(10) && validity(11)
        eyeY = mean([keypoints(2, 2), keypoints(3, 2)]);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);
        if (leftWrist(2) < eyeY + 25 && leftWrist(2) > eyeY - 25) || (rightWrist(2) < eyeY + 25 && rightWrist(2) > eyeY - 25)
            isPhone = true;
        end
    end
end

function isTyping = isTypingPose(keypoints, validity)
    isTyping = false;
    if all(validity([6, 7, 8, 9, 10, 11, 12, 13]))
        shoulderY = mean([keypoints(6, 2), keypoints(7, 2)]);
        hipY = mean([keypoints(12, 2), keypoints(13, 2)]);
        leftElbow = keypoints(8, :); rightElbow = keypoints(9, :);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        typingLevel = shoulderY + (hipY - shoulderY) * 0.7;
        bothHandsAtLevel = abs(leftWrist(2) - typingLevel) < 30 && abs(rightWrist(2) - typingLevel) < 30;
        elbowsPositioned = leftElbow(2) < shoulderY + 20 && rightElbow(2) < shoulderY + 20;

        if bothHandsAtLevel && elbowsPositioned, isTyping = true; end
    end
end

function isFighting = isFightingStance(keypoints, validity)
    isFighting = false;
    if all(validity([6, 7, 8, 9, 10, 11]))
        shoulderY = mean([keypoints(6, 2), keypoints(7, 2)]);
        leftElbow = keypoints(8, :); rightElbow = keypoints(9, :);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        elbowsUp = leftElbow(2) < shoulderY && rightElbow(2) < shoulderY;
        wristsUp = leftWrist(2) < shoulderY && rightWrist(2) < shoulderY;
        guardsUp = abs(leftWrist(1) - rightWrist(1)) > 50;

        if elbowsUp && wristsUp && guardsUp, isFighting = true; end
    end
end

function isWriting = isWritingPose(keypoints, validity)
    isWriting = false;
    if all(validity([6, 7, 10, 11, 12, 13]))
        shoulderY = mean([keypoints(6, 2), keypoints(7, 2)]);
        hipY = mean([keypoints(12, 2), keypoints(13, 2)]);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        writingLevel = shoulderY + (hipY - shoulderY) * 0.65;
        oneHandWriting = abs(leftWrist(2) - writingLevel) < 35 || abs(rightWrist(2) - writingLevel) < 35;

        if oneHandWriting, isWriting = true; end
    end
end

function isHandsUp = isHandsUpPose(keypoints, validity)
    isHandsUp = false;
    if all(validity([6, 7, 10, 11]))
        shoulderY = mean([keypoints(6, 2), keypoints(7, 2)]);
        leftWristY = keypoints(10, 2); rightWristY = keypoints(11, 2);

        if leftWristY < shoulderY - 50 && rightWristY < shoulderY - 50, isHandsUp = true; end
    end
end

function isVictory = isVictoryPose(keypoints, validity)
    isVictory = false;
    if all(validity([6, 7, 10, 11]))
        leftShoulder = keypoints(6, :); rightShoulder = keypoints(7, :);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        shoulderY = mean([leftShoulder(2), rightShoulder(2)]);
        wristsUp = leftWrist(2) < shoulderY - 40 && rightWrist(2) < shoulderY - 40;
        wristsApart = abs(leftWrist(1) - rightWrist(1)) > abs(leftShoulder(1) - rightShoulder(1)) * 1.3;

        if wristsUp && wristsApart, isVictory = true; end
    end
end

function isCrossed = isCrossedArmsPose(keypoints, validity)
    isCrossed = false;
    if all(validity([6, 7, 8, 9]))
        leftShoulder = keypoints(6, :); rightShoulder = keypoints(7, :);
        leftElbow = keypoints(8, :); rightElbow = keypoints(9, :);

        centerX = mean([leftShoulder(1), rightShoulder(1)]);
        if leftElbow(1) > centerX + 10 && rightElbow(1) < centerX - 10, isCrossed = true; end
    end
end

function isPointing = isPointingPose(keypoints, validity)
    isPointing = false;
    if all(validity([6, 7, 8, 9, 10, 11]))
        leftShoulder = keypoints(6, :); rightShoulder = keypoints(7, :);
        leftElbow = keypoints(8, :); rightElbow = keypoints(9, :);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        leftExtended = norm(leftWrist - leftShoulder) > norm(leftElbow - leftShoulder) * 1.6;
        rightExtended = norm(rightWrist - rightShoulder) > norm(rightElbow - rightShoulder) * 1.6;

        if leftExtended || rightExtended, isPointing = true; end
    end
end

function isClapping = isClappingPose(keypoints, validity)
    isClapping = false;
    if validity(10) && validity(11)
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);
        if norm(leftWrist - rightWrist) < 20, isClapping = true; end
    end
end

function isReading = isReadingPose(keypoints, validity)
    isReading = false;
    if all(validity([1, 6, 7, 10, 11]))
        noseY = keypoints(1, 2); shoulderY = mean([keypoints(6, 2), keypoints(7, 2)]);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        headDown = noseY > shoulderY - 20;
        handsInReadingPosition = (leftWrist(2) > shoulderY - 30 && leftWrist(2) < shoulderY + 50) || ...
                                (rightWrist(2) > shoulderY - 30 && rightWrist(2) < shoulderY + 50);

        if headDown && handsInReadingPosition, isReading = true; end
    end
end

function isThinking = isThinkingPose(keypoints, validity)
    isThinking = false;
    if validity(1) && validity(10) && validity(11)
        nosePos = keypoints(1, :); leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        handNearFace = norm(leftWrist - nosePos) < 60 || norm(rightWrist - nosePos) < 60;
        if handNearFace, isThinking = true; end
    end
end

function isSuspicious = isSuspiciousActivity(keypoints, validity, history)
    isSuspicious = false;
    if length(history) < 5, return; end

    try
        recentPoses = history(end-4:end);
        movements = [];
        for i = 1:length(recentPoses)-1
            if all(recentPoses(i).validity([6,7])) && all(recentPoses(i+1).validity([6,7]))
                movement = norm(mean(recentPoses(i+1).keypoints([6,7],:)) - mean(recentPoses(i).keypoints([6,7],:)));
                movements = [movements, movement];
            end
        end

        if ~isempty(movements)
            avgMovement = mean(movements);
            movementVar = var(movements);

            if avgMovement > 10 && movementVar > 50, isSuspicious = true; end
        end
    catch
    end
end

function isCarrying = isCarryingObject(keypoints, validity)
    isCarrying = false;
    if all(validity([6, 7, 8, 9, 10, 11]))
        leftShoulder = keypoints(6, :); rightShoulder = keypoints(7, :);
        leftElbow = keypoints(8, :); rightElbow = keypoints(9, :);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        shoulderWidth = norm(leftShoulder - rightShoulder);
        armSpan = norm(leftWrist - rightWrist);

        if armSpan < shoulderWidth * 0.7, isCarrying = true; end
    end
end

function isStretching = isStretchingPose(keypoints, validity)
    isStretching = false;
    if all(validity([6, 7, 10, 11]))
        leftShoulder = keypoints(6, :); rightShoulder = keypoints(7, :);
        leftWrist = keypoints(10, :); rightWrist = keypoints(11, :);

        leftArmExtended = norm(leftWrist - leftShoulder) > 80;
        rightArmExtended = norm(rightWrist - rightShoulder) > 80;

        if leftArmExtended || rightArmExtended, isStretching = true; end
    end
end

function isSitting = isSittingPose(keypoints, validity)
    isSitting = false;
    if all(validity([12, 13, 14, 15]))
        hipY = mean([keypoints(12, 2), keypoints(13, 2)]);
        kneeY = mean([keypoints(14, 2), keypoints(15, 2)]);

        if abs(hipY - kneeY) < 35, isSitting = true; end
    end
end

function isCrouching = isCrouchingPose(keypoints, validity)
    isCrouching = false;
    if all(validity([6, 7, 12, 13, 14, 15]))
        shoulderY = mean([keypoints(6, 2), keypoints(7, 2)]);
        hipY = mean([keypoints(12, 2), keypoints(13, 2)]);
        kneeY = mean([keypoints(14, 2), keypoints(15, 2)]);

        lowPosture = (hipY - shoulderY) < 80;
        kneesBelow = kneeY > hipY + 20;

        if lowPosture && kneesBelow, isCrouching = true; end
    end
end

function [velocity, verticalVar, armMovement, spatialMovement] = extractAdvancedMovementFeatures(poseHistory)
    velocity = 0; verticalVar = 0; armMovement = 0; spatialMovement = 0;

    if length(poseHistory) < 3, return; end

    hipPositions = []; shoulderPositions = []; wristPositions = [];

    for i = 1:length(poseHistory)
        pose = poseHistory(i); keypoints = pose.keypoints; validity = pose.validity;

        if validity(12) && validity(13)
            hipCenter = mean([keypoints(12, :); keypoints(13, :)], 1);
            hipPositions = [hipPositions; hipCenter];
        end

        if validity(6) && validity(7)
            shoulderCenter = mean([keypoints(6, :); keypoints(7, :)], 1);
            shoulderPositions = [shoulderPositions; shoulderCenter];
        end

        if validity(10) && validity(11)
            wristCenter = mean([keypoints(10, :); keypoints(11, :)], 1);
            wristPositions = [wristPositions; wristCenter];
        end
    end

    if size(hipPositions, 1) >= 2
        hipMovement = diff(hipPositions, 1, 1);
        velocity = mean(sqrt(sum(hipMovement.^2, 2)));
        verticalVar = var(hipPositions(:, 2));
    end

    if size(wristPositions, 1) >= 2
        wristMovement = diff(wristPositions, 1, 1);
        armMovement = mean(sqrt(sum(wristMovement.^2, 2)));
    end

    if size(shoulderPositions, 1) >= 2
        shoulderMovement = diff(shoulderPositions, 1, 1);
        spatialMovement = mean(sqrt(sum(shoulderMovement.^2, 2)));
    end
end

function securityAssessment = assessSecurityRisk(activity, confidence, biomechanics, frameCount)
    securityAssessment = struct('level', 'Low', 'details', 'Normal behavior');

    highRiskActivities = {'Fighting_Stance', 'Suspicious_Activity', 'Hands_Up', 'Aggressive_Posture'};
    mediumRiskActivities = {'Loitering', 'Looking_Around', 'Carrying_Object', 'Crouching'};

    if any(strcmp(activity, highRiskActivities))
        if confidence > 0.8
            securityAssessment.level = 'Critical';
            securityAssessment.details = sprintf('High confidence %s detected', activity);
        else
            securityAssessment.level = 'High';
            securityAssessment.details = sprintf('Potential %s activity', activity);
        end
    elseif any(strcmp(activity, mediumRiskActivities))
        securityAssessment.level = 'Medium';
        securityAssessment.details = sprintf('%s behavior observed', activity);
    elseif strcmp(biomechanics.posture, 'Poor') && confidence > 0.7
        securityAssessment.level = 'Low';
        securityAssessment.details = 'Poor posture detected';
    end
end

function color = getAdvancedActivityColor(activity)
    switch activity
        case {'Drinking_Water', 'Drinking_Coffee', 'Eating', 'Phone_Call', 'Texting'}
            color = 'blue';
        case {'Fighting_Stance', 'Hands_Up', 'Suspicious_Activity', 'Aggressive_Posture'}
            color = 'red';
        case {'Victory_Pose', 'Clapping', 'Waving', 'Thumbs_Up', 'Greeting', 'High_Five'}
            color = 'green';
        case {'Running', 'Jogging', 'Jumping', 'Exercise_Pose', 'Stretching', 'Yoga_Pose'}
            color = 'magenta';
        case {'Sitting', 'Standing', 'Good_Posture', 'Meditation', 'Relaxing'}
            color = 'cyan';
        case {'Writing', 'Typing', 'Reading', 'Studying', 'Working_Computer'}
            color = 'blue';
        case {'Pointing', 'Teaching', 'Presentation', 'Explaining', 'Demonstrating'}
            color = 'yellow';
        case {'Poor_Posture', 'Slouching', 'Fidgeting', 'Restless'}
            color = 'red';
        otherwise
            color = 'black';
    end
end

function closestFace = findClosestFace(humanBbox, faceResults)
    closestFace = [];
    if isempty(faceResults), return; end

    humanCenter = [humanBbox(1) + humanBbox(3)/2, humanBbox(2) + humanBbox(4)/2];
    minDistance = inf;

    for i = 1:length(faceResults)
        faceCenter = [faceResults(i).bbox(1) + faceResults(i).bbox(3)/2, ...
                     faceResults(i).bbox(2) + faceResults(i).bbox(4)/2];
        distance = norm(humanCenter - faceCenter);

        if distance < minDistance && distance < 150
            minDistance = distance;
            closestFace = faceResults(i);
        end
    end
end

function frame = insert3DPoseVisualization(frame, keypoints, validity, scores)
    try
        colors = {'red', 'green', 'blue', 'yellow', 'magenta', 'cyan'};

        for i = 1:size(keypoints, 1)
            if validity(i)
                confidence = scores(i);
                markerSize = max(3, round(confidence * 8));
                colorIdx = mod(i-1, length(colors)) + 1;

                frame = insertMarker(frame, keypoints(i, :), 'o', ...
                    'Color', colors{colorIdx}, 'Size', markerSize);
            end
        end

        connections = [
            1, 6; 1, 7; 6, 8; 7, 9; 8, 10; 9, 11;
            6, 12; 7, 13; 12, 14; 13, 15; 14, 16; 15, 17;
            12, 13; 6, 7
        ];

        for i = 1:size(connections, 1)
            pt1 = connections(i, 1); pt2 = connections(i, 2);
            if validity(pt1) && validity(pt2)
                avgConfidence = (scores(pt1) + scores(pt2)) / 2;
                lineWidth = max(1, round(avgConfidence * 4));

                frame = insertShape(frame, 'Line', ...
                    [keypoints(pt1, :), keypoints(pt2, :)], ...
                    'Color', 'green', 'LineWidth', lineWidth);
            end
        end

    catch
        frame = insertObjectKeypoints(frame, keypoints, 'KeypointVisibility', validity, ...
            'ConnectionColor', 'green', 'KeypointColor', 'yellow', 'KeypointSize', 4, 'LineWidth', 2);
    end
end

function updateIndividualLog(individualLogs, personID, personName, activity, confidence, frameCount)
    try
        logKey = sprintf('person_%d', personID);
        timestamp = now;

        if isKey(individualLogs, logKey)
            log = individualLogs(logKey);
        else
            log = struct('personID', personID, 'name', personName, 'activities', [], 'totalFrames', 0, 'firstSeen', timestamp);
        end

        log.activities(end+1) = struct('activity', activity, 'confidence', confidence, ...
                                      'timestamp', timestamp, 'frame', frameCount);
        log.totalFrames = log.totalFrames + 1;
        log.lastSeen = timestamp;

        if length(log.activities) > 1000
            log.activities = log.activities(end-999:end);
        end

        individualLogs(logKey) = log;
    catch ME
        fprintf('Individual log error: %s\n', ME.message);
    end
end

function initializeCSVLog(filename)
    try
        fid = fopen(filename, 'w');
        fprintf(fid, 'Timestamp,Frame,PersonID,PersonName,Activity,Confidence,Posture,Balance,Energy,Coordination\n');
        fclose(fid);
    catch ME
        fprintf('CSV initialization error: %s\n', ME.message);
    end
end

function logToCSV(filename, frameCount, personID, personName, activity, confidence, biomechanics)
    try
        fid = fopen(filename, 'a');
        fprintf(fid, '%s,%d,%d,%s,%s,%.3f,%s,%.3f,%.3f,%.3f\n', ...
                datestr(now, 'yyyy-mm-dd HH:MM:SS'), frameCount, personID, personName, ...
                activity, confidence, biomechanics.posture, biomechanics.balance, ...
                biomechanics.energy, biomechanics.coordination);
        fclose(fid);
    catch ME
        fprintf('CSV logging error: %s\n', ME.message);
    end
end

function updateUserInterface(hPeopleList, hActivityList, hSecurityList, hSystemStats, hSecurityLevel, ...
                            currentPeople, currentActivities, currentSecurity, frameCount, ...
                            totalPeopleDetected, totalActivitiesLogged, securityAlertLevel, avgProcessTime)
    try
        if ~isempty(currentPeople)
            set(hPeopleList, 'String', currentPeople);
        end

        if ~isempty(currentActivities)
            set(hActivityList, 'String', currentActivities);
        end

        if ~isempty(currentSecurity)
            set(hSecurityList, 'String', currentSecurity);
        end

        statsText = sprintf(['Frame: %d\nFPS: %.1f\nTotal People: %d\nActivities Logged: %d\n' ...
                            'Processing Time: %.3fs\nMemory Usage: Active\nSystem Status: Running'], ...
                           frameCount, 1/max(avgProcessTime,0.01), totalPeopleDetected, ...
                           totalActivitiesLogged, avgProcessTime);
        set(hSystemStats, 'String', statsText);

        securityText = sprintf('Security Level: %s', securityAlertLevel);
        if strcmp(securityAlertLevel, 'Critical') || strcmp(securityAlertLevel, 'High')
            set(hSecurityLevel, 'BackgroundColor', [1.0 0.8 0.8]);
        elseif strcmp(securityAlertLevel, 'Medium')
            set(hSecurityLevel, 'BackgroundColor', [1.0 1.0 0.8]);
        else
            set(hSecurityLevel, 'BackgroundColor', [0.8 1.0 0.8]);
        end
        set(hSecurityLevel, 'String', securityText);

    catch ME
        fprintf('UI update error: %s\n', ME.message);
    end
end

function exportFinalReport(individualLogs, securityEvents, sessionStartTime, frameCount, csvFile)
    try
        reportFile = sprintf('session_report_%s.txt', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
        fid = fopen(reportFile, 'w');

        fprintf(fid, '=== COMPREHENSIVE FACE + POSTURE ANALYSIS REPORT ===\n');
        fprintf(fid, 'Session Start: %s\n', datestr(sessionStartTime));
        fprintf(fid, 'Session End: %s\n', datestr(now));
        fprintf(fid, 'Duration: %.1f minutes\n', (now - sessionStartTime) * 24 * 60);
        fprintf(fid, 'Total Frames: %d\n\n', frameCount);

        fprintf(fid, '=== INDIVIDUAL ANALYSIS ===\n');
        logKeys = individualLogs.keys;
        for i = 1:length(logKeys)
            log = individualLogs(logKeys{i});
            fprintf(fid, 'Person ID%d (%s):\n', log.personID, log.name);
            fprintf(fid, '  Total Frames: %d\n', log.totalFrames);
            fprintf(fid, '  First Seen: %s\n', datestr(log.firstSeen));
            fprintf(fid, '  Last Seen: %s\n', datestr(log.lastSeen));

            if ~isempty(log.activities)
                activities = {log.activities.activity};
                [uniqueActs, ~, idx] = unique(activities);
                counts = accumarray(idx, 1);

                fprintf(fid, '  Top Activities:\n');
                [~, sortIdx] = sort(counts, 'descend');
                for j = 1:min(5, length(sortIdx))
                    actIdx = sortIdx(j);
                    fprintf(fid, '    %s: %d times\n', uniqueActs{actIdx}, counts(actIdx));
                end
            end
            fprintf(fid, '\n');
        end

        fprintf(fid, '=== SECURITY EVENTS ===\n');
        fprintf(fid, 'Total Security Events: %d\n', length(securityEvents));
        for i = 1:length(securityEvents)
            event = securityEvents(i);
            fprintf(fid, '%s - ID%d: %s (%s) - %s\n', datestr(event.timestamp), ...
                    event.personID, event.activity, event.risk, event.details);
        end

        fprintf(fid, '\n=== DATA FILES ===\n');
        fprintf(fid, 'Detailed CSV Log: %s\n', csvFile);
        fprintf(fid, 'Session Report: %s\n', reportFile);

        fclose(fid);
        fprintf('Final report exported: %s\n', reportFile);

    catch ME
        fprintf('Report export error: %s\n', ME.message);
    end
end

function result = iff(condition, trueValue, falseValue)
    if condition, result = trueValue; else, result = falseValue; end
end

function mainCloseCallback(src, ~)
    try
        set(src, 'CloseRequestFcn', '');
        delete(src);
    catch
    end
end

function learnActivityCallback(~, ~)
    fprintf('Activity learning mode activated\n');
end

function exportCSVCallback(~, ~)
    fprintf('CSV export requested\n');
end
