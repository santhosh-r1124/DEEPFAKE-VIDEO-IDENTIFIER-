/**
 * NOXIS — Detection Controller v2.1
 * Upload → API → results + frame-level slider with dynamic Grad-CAM viewer.
 */
(function () {
    'use strict';

    var $ = function (id) { return document.getElementById(id); };

    /* ── DOM ──────────────────────────────────────────────── */
    var uploadZone = $('uploadZone'),
        fileInput = $('fileInput'),
        fileInfo = $('fileInfo'),
        fileName = $('fileName'),
        fileSize = $('fileSize'),
        fileRemove = $('fileRemove'),
        scanSection = $('scanSection'),
        scanBtn = $('scanBtn'),
        uploadSection = $('uploadSection'),
        analysisOverlay = $('analysisOverlay'),
        analysisStatus = $('analysisStatus'),
        analysisProgress = $('analysisProgress'),
        resultsPanel = $('resultsPanel'),
        resultsHeader = $('resultsHeader'),
        verdictLabel = $('verdictLabel'),
        verdictBadge = $('verdictBadge'),
        confidenceValue = $('confidenceValue'),
        statPrediction = $('statPrediction'),
        statConfidence = $('statConfidence'),
        statProbability = $('statProbability'),
        statTime = $('statTime'),
        probFill = $('probFill'),
        probPercentage = $('probPercentage'),
        frameSection = $('frameSection'),
        frameCounter = $('frameCounter'),
        frameImage = $('frameImage'),
        frameBadge = $('frameBadge'),
        frameProbValue = $('frameProbValue'),
        frameProbFill = $('frameProbFill'),
        frameVerdict = $('frameVerdict'),
        frameSlider = $('frameSlider'),
        frameMinimap = $('frameMinimap'),
        gradcamSection = $('gradcamSection'),
        gradcamTrack = $('gradcamTrack'),
        newScanBtn = $('newScanBtn'),
        lightbox = $('lightbox'),
        lightboxImg = $('lightboxImg'),
        lightboxClose = $('lightboxClose'),
        errorBanner = $('errorBanner'),
        errorText = $('errorText'),
        errorClose = $('errorClose');

    var selectedFile = null;
    var frameData = [];  // Per-frame predictions from API
    var API = window.location.origin;

    /* ── Helpers ──────────────────────────────────────────── */
    function fmtBytes(b) {
        if (b < 1024) return b + ' B';
        if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
        return (b / 1048576).toFixed(1) + ' MB';
    }

    function showError(msg) { errorText.textContent = msg; errorBanner.classList.add('visible'); }
    function hideError() { errorBanner.classList.remove('visible'); }
    errorClose.addEventListener('click', hideError);

    function riskLevel(p) {
        if (p < 0.35) return 'low';
        if (p < 0.65) return 'mid';
        return 'high';
    }

    /* ── Drag & Drop ─────────────────────────────────────── */
    uploadZone.addEventListener('click', function () { fileInput.click(); });
    uploadZone.addEventListener('dragover', function (e) { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', function () { uploadZone.classList.remove('dragover'); });
    uploadZone.addEventListener('drop', function (e) {
        e.preventDefault(); uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        hideError();
        var ok = file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i) || file.type.indexOf('video') === 0;
        if (!ok) { showError('Unsupported format. Use MP4, AVI, MOV, MKV or WebM.'); return; }
        if (file.size > 200 * 1024 * 1024) { showError('File too large (max 200 MB).'); return; }
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = fmtBytes(file.size);
        fileInfo.classList.add('visible');
        scanSection.style.display = 'block';
    }

    fileRemove.addEventListener('click', function () {
        selectedFile = null; fileInput.value = '';
        fileInfo.classList.remove('visible');
        scanSection.style.display = 'none';
        hideError();
    });

    /* ── Scan ────────────────────────────────────────────── */
    scanBtn.addEventListener('click', function () {
        if (!selectedFile) return;
        startScan();
    });

    var statusMsgs = [
        'Detecting faces in frames\u2026',
        'Extracting spatial features\u2026',
        'Computing FFT frequency analysis\u2026',
        'Running temporal sequence model\u2026',
        'Generating Grad-CAM heatmaps\u2026',
        'Computing per-frame probabilities\u2026',
        'Finalizing report\u2026',
    ];

    function startScan() {
        hideError();
        uploadSection.style.display = 'none';
        resultsPanel.classList.remove('visible');
        analysisOverlay.classList.add('active');
        analysisProgress.style.width = '0%';

        var mi = 0, prog = 0;
        var msgIv = setInterval(function () { mi = (mi + 1) % statusMsgs.length; analysisStatus.textContent = statusMsgs[mi]; }, 2000);
        var progIv = setInterval(function () { prog = Math.min(prog + Math.random() * 4 + 1, 88); analysisProgress.style.width = prog + '%'; }, 500);

        var fd = new FormData();
        fd.append('video', selectedFile);

        fetch(API + '/predict', { method: 'POST', body: fd })
            .then(function (r) {
                if (!r.ok) return r.json().then(function (d) { return Promise.reject(d); });
                return r.json();
            })
            .then(function (data) {
                clearInterval(msgIv); clearInterval(progIv);
                analysisProgress.style.width = '100%';
                analysisStatus.textContent = 'Complete.';
                setTimeout(function () { analysisOverlay.classList.remove('active'); render(data); }, 500);
            })
            .catch(function (err) {
                clearInterval(msgIv); clearInterval(progIv);
                analysisOverlay.classList.remove('active');
                uploadSection.style.display = 'block';
                showError(err.error || err.message || 'Analysis failed.');
            });
    }

    /* ── Render results ──────────────────────────────────── */
    function render(d) {
        var fake = d.prediction === 'FAKE';
        var pct = (d.probability * 100).toFixed(1);
        var conf = (d.confidence * 100).toFixed(1);

        resultsHeader.className = 'results-header ' + (fake ? 'fake-bg' : 'real-bg');
        verdictLabel.textContent = d.prediction;
        verdictLabel.className = 'verdict-label ' + (fake ? 'fake' : 'real');
        verdictBadge.textContent = d.prediction;
        verdictBadge.className = 'badge ' + (fake ? 'badge-fake' : 'badge-real');
        confidenceValue.textContent = conf + '%';
        confidenceValue.style.color = fake ? 'var(--red)' : 'var(--green)';

        statPrediction.textContent = d.prediction;
        statPrediction.style.color = fake ? 'var(--red)' : 'var(--green)';
        statConfidence.textContent = conf + '%';
        statProbability.textContent = pct + '%';
        statTime.textContent = (d.inference_time || '\u2014') + 's';

        setTimeout(function () {
            probFill.style.width = pct + '%';
            probFill.className = 'fill ' + (fake ? 'fake' : 'real');
        }, 80);
        probPercentage.textContent = pct + '% Manipulated';
        probPercentage.style.color = fake ? 'var(--red)' : 'var(--green)';

        /* ── Frame-level predictions ─────────────────────── */
        frameData = d.frame_predictions || [];

        if (frameData.length > 0) {
            frameSection.style.display = 'block';
            frameSlider.max = frameData.length - 1;
            frameSlider.value = 0;

            buildMinimap();
            updateFrameView(0);

            frameSlider.addEventListener('input', function () {
                updateFrameView(parseInt(this.value));
            });
        } else {
            frameSection.style.display = 'none';
        }

        /* ── Grad-CAM gallery ────────────────────────────── */
        gradcamTrack.innerHTML = '';
        var heatmaps = d.heatmap_frames || [];
        if (heatmaps.length > 0) {
            gradcamSection.style.display = 'block';
            heatmaps.forEach(function (path, i) {
                var frame = document.createElement('div');
                frame.className = 'gradcam-frame';
                var img = document.createElement('img');
                img.src = path; img.alt = 'Frame ' + (i + 1); img.loading = 'lazy';
                var label = document.createElement('div');
                label.className = 'gradcam-frame-label';
                label.textContent = 'FRAME ' + (i + 1);
                frame.appendChild(img); frame.appendChild(label);
                frame.addEventListener('click', function () {
                    lightboxImg.src = path; lightbox.classList.add('active');
                });
                gradcamTrack.appendChild(frame);
            });
        } else {
            gradcamSection.style.display = 'none';
        }

        resultsPanel.classList.add('visible');
    }

    /* ── Frame viewer ────────────────────────────────────── */
    function updateFrameView(idx) {
        if (idx >= frameData.length) return;
        var f = frameData[idx];
        var p = f.probability;
        var pPct = (p * 100).toFixed(1);
        var isFake = p >= 0.5;
        var risk = riskLevel(p);

        frameCounter.textContent = 'Frame ' + (idx + 1) + ' / ' + frameData.length;

        // Image
        if (f.heatmap) {
            frameImage.src = f.heatmap;
        } else {
            frameImage.src = '';
        }

        // Badge
        frameBadge.textContent = isFake ? 'FAKE' : 'REAL';
        frameBadge.className = 'frame-badge ' + (isFake ? 'fake' : 'real');

        // Probability
        frameProbValue.textContent = pPct + '%';
        frameProbValue.style.color = isFake ? 'var(--red)' : 'var(--green)';
        frameProbFill.style.width = pPct + '%';
        frameProbFill.className = 'frame-prob-fill ' + risk;

        // Verdict
        frameVerdict.textContent = isFake ? 'Manipulated' : 'Authentic';
        frameVerdict.className = 'frame-verdict ' + (isFake ? 'fake' : 'real');

        // Highlight minimap
        var bars = frameMinimap.children;
        for (var i = 0; i < bars.length; i++) {
            bars[i].classList.toggle('active', i === idx);
        }
    }

    function buildMinimap() {
        frameMinimap.innerHTML = '';
        frameData.forEach(function (f, i) {
            var bar = document.createElement('div');
            bar.className = 'minimap-bar ' + riskLevel(f.probability);
            bar.title = 'Frame ' + (i + 1) + ': ' + (f.probability * 100).toFixed(1) + '%';
            bar.addEventListener('click', function () {
                frameSlider.value = i;
                updateFrameView(i);
            });
            frameMinimap.appendChild(bar);
        });
    }

    /* ── Lightbox ─────────────────────────────────────────── */
    lightboxClose.addEventListener('click', function () { lightbox.classList.remove('active'); });
    lightbox.addEventListener('click', function (e) { if (e.target === lightbox) lightbox.classList.remove('active'); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') lightbox.classList.remove('active'); });

    // Click frame image to open lightbox
    frameImage.addEventListener('click', function () {
        if (frameImage.src) { lightboxImg.src = frameImage.src; lightbox.classList.add('active'); }
    });

    /* ── New scan ─────────────────────────────────────────── */
    newScanBtn.addEventListener('click', function () {
        selectedFile = null; fileInput.value = ''; frameData = [];
        fileInfo.classList.remove('visible');
        scanSection.style.display = 'none';
        resultsPanel.classList.remove('visible');
        analysisOverlay.classList.remove('active');
        analysisProgress.style.width = '0%';
        probFill.style.width = '0%';
        gradcamTrack.innerHTML = '';
        frameMinimap.innerHTML = '';
        frameSection.style.display = 'none';
        uploadSection.style.display = 'block';
        hideError();
    });

})();
