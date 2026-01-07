// Upload functionality
let uploadedFiles = [];
let modelLoaded = false;

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    // Check if model is loaded from global variable
    if (typeof window.modelLoaded !== 'undefined') {
        modelLoaded = window.modelLoaded;
    }

    setupEventListeners();
    updateConfidenceDisplay();
});

function setupEventListeners() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const loadModelBtn = document.getElementById('load-model-btn');
    const processBtn = document.getElementById('process-btn');
    const clearBtn = document.getElementById('clear-btn');
    const confSlider = document.getElementById('conf-threshold');

    // Drag and drop
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }

    // File input change
    if (fileInput) {
        fileInput.addEventListener('change', function (e) {
            handleFileSelect(e);
        });
    }

    // Load model button
    if (loadModelBtn) {
        loadModelBtn.addEventListener('click', loadModel);
    }

    // Process button
    if (processBtn) {
        processBtn.addEventListener('click', processImages);
    }

    // Clear button
    if (clearBtn) {
        clearBtn.addEventListener('click', clearFiles);
    }

    // Confidence slider
    if (confSlider) {
        confSlider.addEventListener('input', updateConfidenceDisplay);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('upload-area').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('upload-area').classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
        handleFiles(Array.from(files));
    }
}

function handleFileSelect(e) {
    const files = e.target && e.target.files ? e.target.files : null;
    if (files && files.length > 0) {
        handleFiles(Array.from(files));
    }
}

function handleFiles(files) {
    if (!files || files.length === 0) {
        return;
    }

    const imageFiles = Array.from(files).filter(file => {
        if (!file.name || !file.type) {
            return false;
        }
        const ext = file.name.split('.').pop().toLowerCase();
        const validExts = ['jpg', 'jpeg', 'png'];
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        return validExts.includes(ext) || validTypes.includes(file.type);
    });

    if (imageFiles.length === 0) {
        showToast('Please select valid image files (JPG, PNG)', 'error');
        return;
    }

    if (imageFiles.length < files.length) {
        showToast(`Added ${imageFiles.length} valid image(s). ${files.length - imageFiles.length} invalid file(s) skipped.`, 'info');
    }

    // Check for duplicates
    const existingNames = uploadedFiles.map(f => f.name.toLowerCase());
    const newFiles = imageFiles.filter(file => !existingNames.includes(file.name.toLowerCase()));

    if (newFiles.length < imageFiles.length) {
        showToast(`${imageFiles.length - newFiles.length} duplicate file(s) skipped.`, 'info');
    }

    if (newFiles.length > 0) {
        uploadedFiles = [...uploadedFiles, ...newFiles];
        updateFileList();
        updatePreview();
        showToast(`Added ${newFiles.length} image(s)`, 'success');
    }

    const processBtn = document.getElementById('process-btn');
    if (processBtn) {
        processBtn.disabled = !modelLoaded || uploadedFiles.length === 0;
    }
}

function updateFileList() {
    const fileList = document.getElementById('file-list');
    const fileListContainer = document.getElementById('file-list-container');
    const fileCount = document.getElementById('file-count');

    if (!fileList || !fileListContainer) return;

    fileList.innerHTML = '';

    if (uploadedFiles.length === 0) {
        fileListContainer.style.display = 'none';
        if (fileCount) fileCount.textContent = '0';
        return;
    }

    // Show container and update count
    fileListContainer.style.display = 'block';
    if (fileCount) fileCount.textContent = uploadedFiles.length;

    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span class="file-name" title="${file.name}">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button onclick="removeFile(${index})" class="remove-file-btn" title="Remove file">✕</button>
        `;
        fileList.appendChild(fileItem);
    });
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    updateFileList();
    updatePreview();
    document.getElementById('process-btn').disabled = !modelLoaded || uploadedFiles.length === 0;
}

function updatePreview() {
    const preview = document.getElementById('upload-preview');
    preview.innerHTML = '';

    uploadedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const item = document.createElement('div');
            item.className = 'preview-item';
            item.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <div class="preview-name">${file.name}</div>
            `;
            preview.appendChild(item);
        };
        reader.readAsDataURL(file);
    });
}

function updateConfidenceDisplay() {
    const slider = document.getElementById('conf-threshold');
    const value = document.getElementById('conf-value');
    if (slider && value) {
        value.textContent = parseFloat(slider.value).toFixed(2);
    }
}

async function clearFiles() {
    uploadedFiles = [];
    const fileList = document.getElementById('file-list');
    const fileListContainer = document.getElementById('file-list-container');
    const uploadPreview = document.getElementById('upload-preview');
    const fileInput = document.getElementById('file-input');
    const processBtn = document.getElementById('process-btn');

    if (fileList) fileList.innerHTML = '';
    if (fileListContainer) fileListContainer.style.display = 'none';
    if (uploadPreview) uploadPreview.innerHTML = '';
    if (fileInput) fileInput.value = '';
    if (processBtn) processBtn.disabled = true;

    updateFileList();

    // Call backend to clear session files
    try {
        await fetch('/api/clear', { method: 'POST' });
        showToast('Files cleared', 'info');
    } catch (e) {
        console.error("Error clearing backend files:", e);
        showToast('Files cleared (local only)', 'info');
    }
}

async function loadModel() {
    const btn = document.getElementById('load-model-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Loading...';

    try {
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_path: document.getElementById('model-path').value
            })
        });

        const data = await response.json();

        if (data.success) {
            modelLoaded = true;
            btn.textContent = '✅ Model Loaded';
            btn.disabled = true;
            showToast('Model loaded successfully!', 'success');
            document.getElementById('process-btn').disabled = uploadedFiles.length === 0;
        } else {
            showToast(data.message || 'Failed to load model', 'error');
            btn.disabled = false;
            btn.textContent = '🚀 Load Model';
        }
    } catch (error) {
        showToast('Error loading model: ' + error.message, 'error');
        btn.disabled = false;
        btn.textContent = '🚀 Load Model';
    }
}

async function processImages() {
    if (!modelLoaded) {
        showToast('Please load the model first', 'error');
        return;
    }

    if (uploadedFiles.length === 0) {
        showToast('Please upload images first', 'error');
        return;
    }

    const processBtn = document.getElementById('process-btn');
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    processBtn.disabled = true;
    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading files...';

    // Upload files
    const formData = new FormData();
    uploadedFiles.forEach(file => {
        formData.append('files[]', file);
    });

    try {
        // Upload files
        let uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Failed to upload files');
        }

        const uploadData = await uploadResponse.json();
        progressFill.style.width = '30%';
        progressText.textContent = 'Processing images...';

        // Process images
        const processResponse = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conf_threshold: parseFloat(document.getElementById('conf-threshold').value),
                img_size: parseInt(document.getElementById('img-size').value)
            })
        });

        if (!processResponse.ok) {
            throw new Error('Failed to process images');
        }

        progressFill.style.width = '100%';
        progressText.textContent = '✅ Processing complete!';

        const processData = await processResponse.json();

        if (processData.success) {
            showToast(`Successfully processed ${processData.summary.processed_count} images!`, 'success');
            setTimeout(() => {
                window.location.href = '/results';
            }, 1500);
        } else {
            throw new Error(processData.message || 'Processing failed');
        }

    } catch (error) {
        showToast('Error: ' + error.message, 'error');
        processBtn.disabled = false;
        progressContainer.style.display = 'none';
    }
}

