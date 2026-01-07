// Global variables
let selectedFiles = [];
let allResults = [];
let currentProcessingIndex = 0;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const detectBtn = document.getElementById('detectBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const summarySection = document.getElementById('summarySection');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

// Check backend connection on load
async function checkBackendConnection() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Connected to backend';
        } else {
            statusDot.className = 'status-dot disconnected';
            statusText.textContent = 'Backend unavailable';
        }
    } catch (error) {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Connection failed';
    }
}

// Upload area click event
uploadArea.addEventListener('click', () => fileInput.click());

// File input change event
fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Drag and drop events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

// Handle file selection
function handleFiles(files) {
    const fileArray = Array.from(files);
    
    // Filter valid image files
    const validFiles = fileArray.filter(file => {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        return validTypes.includes(file.type);
    });
    
    if (validFiles.length !== fileArray.length) {
        alert('Some files were skipped. Only PNG, JPG, and JPEG files are supported.');
    }
    
    // Sort alphabetically
    validFiles.sort((a, b) => a.name.localeCompare(b.name));
    
    selectedFiles = [...selectedFiles, ...validFiles];
    updateFileList();
    detectBtn.disabled = false;
    clearBtn.disabled = false;
}

// Update file list display
function updateFileList() {
    fileList.innerHTML = '';
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        const fileInfo = document.createElement('div');
        fileInfo.className = 'file-info';
        
        const fileIcon = document.createElement('span');
        fileIcon.className = 'file-icon';
        fileIcon.textContent = '📎';
        
        const fileName = document.createElement('span');
        fileName.className = 'file-name';
        fileName.textContent = file.name;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.textContent = 'Remove';
        removeBtn.onclick = () => removeFile(index);
        
        fileInfo.appendChild(fileIcon);
        fileInfo.appendChild(fileName);
        fileItem.appendChild(fileInfo);
        fileItem.appendChild(removeBtn);
        fileList.appendChild(fileItem);
    });
}

// Remove file from list
function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
    if (selectedFiles.length === 0) {
        detectBtn.disabled = true;
        clearBtn.disabled = true;
    }
}

// Clear all files and results
clearBtn.addEventListener('click', () => {
    selectedFiles = [];
    allResults = [];
    currentProcessingIndex = 0;
    fileList.innerHTML = '';
    resultsContainer.innerHTML = '';
    resultsSection.style.display = 'none';
    detectBtn.disabled = true;
    clearBtn.disabled = true;
    fileInput.value = '';
    
    // Remove view toggle button if exists
    const toggleSection = document.getElementById('viewToggleSection');
    if (toggleSection) {
        toggleSection.remove();
    }
    viewMode = 'expanded';
});

// Detect defects - Progressive Processing
detectBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;

    // Reset
    allResults = [];
    currentProcessingIndex = 0;
    resultsContainer.innerHTML = '';
    
    // Show loading and results section
    loading.style.display = 'block';
    resultsSection.style.display = 'block';
    summarySection.style.display = 'none';
    detectBtn.disabled = true;

    // Start in expanded mode
    addViewToggle();
    viewMode = 'expanded';
    
    const btn = document.getElementById('viewModeBtn');
    if (btn) {
        btn.innerHTML = '🔲 Switch to Grid View';
    }

    // Process images one by one
    for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        
        // Update progress
        const progress = ((i + 1) / selectedFiles.length) * 100;
        progressBar.style.width = progress + '%';
        progressText.textContent = `${i + 1} / ${selectedFiles.length}`;
        loadingText.textContent = `Processing: ${file.name}`;

        try {
            const result = await processImageProgressive(file);
            allResults.push(result);
            
            if (viewMode === 'expanded') {
                displaySingleResult(result, i);
            } else if (viewMode === 'grid') {
                updateGridView();
            }
            
        } catch (error) {
            console.error(`Error processing ${file.name}:`, error);
            allResults.push({
                filename: file.name,
                error: error.message
            });
            
            if (viewMode === 'expanded') {
                displaySingleResult({
                    filename: file.name,
                    error: error.message
                }, i);
            } else if (viewMode === 'grid') {
                updateGridView();
            }
        }
    }

    // Hide loading, show summary
    loading.style.display = 'none';
    displaySummary(allResults);
    detectBtn.disabled = false;
});

// Process single image
async function processImageProgressive(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/detect-progressive', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        let errorData;
        try {
            errorData = await response.json();
        } catch {
            errorData = { error: `Server returned ${response.status}` };
        }
        throw new Error(errorData.error || 'Detection failed');
    }

    return await response.json();
}

// Display single result
function displaySingleResult(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.opacity = '0';
    
    if (result.error) {
        card.innerHTML = `
            <div class="error">Error processing ${result.filename}: ${result.error}</div>
        `;
    } else {
        let detectionsHTML = '';
        if (result.detections && result.detections.length > 0) {
            detectionsHTML = `
                <table class="detections-table">
                    <thead>
                        <tr>
                            <th>Defect Type</th>
                            <th>Confidence</th>
                            <th>X1</th>
                            <th>Y1</th>
                            <th>X2</th>
                            <th>Y2</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${result.detections.map(det => `
                            <tr>
                                <td><strong>${det.defect_type}</strong></td>
                                <td>${(det.confidence * 100).toFixed(2)}%</td>
                                <td>${det.x1}</td>
                                <td>${det.y1}</td>
                                <td>${det.x2}</td>
                                <td>${det.y2}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        } else {
            detectionsHTML = '<p style="text-align: center; color: #4caf50; padding: 20px; font-weight: 600;">✅ No defects detected - PCB looks good!</p>';
        }

        card.innerHTML = `
            <div class="result-header">
                <div class="result-title">📷 ${result.filename}</div>
                <div class="defect-badge">${result.defect_count || 0} Defects</div>
            </div>
            <div class="image-comparison">
                <div class="image-container">
                    <div class="image-label">Original Image</div>
                    <img src="${result.original_image}" alt="Original">
                </div>
                <div class="image-container">
                    <div class="image-label">Detected Defects</div>
                    <img src="${result.annotated_image}" alt="Annotated">
                </div>
            </div>
            ${detectionsHTML}
            <div class="download-buttons">
                <button class="btn btn-download" onclick="downloadImage('${result.filename}', '${result.annotated_image}', 'annotated')">
                    💾 Download Annotated
                </button>
                <button class="btn btn-download" onclick="downloadImage('${result.filename}', '${result.original_image}', 'original')">
                    📥 Download Original
                </button>
                ${result.detections && result.detections.length > 0 ? 
                    `<button class="btn btn-download" onclick='downloadSingleCSV(${JSON.stringify(result.detections)}, "${result.filename}")'>
                        📄 Download CSV
                    </button>` : 
                    '<button class="btn btn-download" disabled>📄 No Data for CSV</button>'
                }
            </div>
        `;
    }
    
    resultsContainer.appendChild(card);
    
    setTimeout(() => {
        card.style.transition = 'opacity 0.5s ease-in';
        card.style.opacity = '1';
    }, 100);
}

// Display summary with chart
function displaySummary(results) {
    summarySection.style.display = 'block';
    
    let totalDefects = 0;
    const defectTypes = {};
    let totalImages = results.length;
    
    results.forEach(result => {
        if (!result.error && result.detections) {
            totalDefects += result.defect_count || 0;
            result.detections.forEach(det => {
                defectTypes[det.defect_type] = (defectTypes[det.defect_type] || 0) + 1;
            });
        }
    });
    
    const summaryGrid = document.getElementById('summaryGrid');
    summaryGrid.innerHTML = `
        <div class="summary-card">
            <div class="summary-value">${totalImages}</div>
            <div class="summary-label">Images Processed</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">${totalDefects}</div>
            <div class="summary-label">Total Defects</div>
        </div>
        <div class="summary-card">
            <div class="summary-value">${Object.keys(defectTypes).length}</div>
            <div class="summary-label">Defect Types</div>
        </div>
    `;
    
    if (Object.keys(defectTypes).length > 0) {
        createDefectChart(defectTypes);
    }
}

// Create defect distribution chart
function createDefectChart(defectTypes) {
    const chartContainer = document.getElementById('chartContainer');
    chartContainer.innerHTML = '<canvas id="defectChart"></canvas>';
    
    const ctx = document.getElementById('defectChart').getContext('2d');
    
    const labels = Object.keys(defectTypes);
    const data = Object.values(defectTypes);
    const colors = generateColors(labels.length);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Number of Defects',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.6', '1')),
                borderWidth: 2,
                barThickness: 50,
                maxBarThickness: 60
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Defect Distribution Across All Images',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    color: '#333',
                    padding: {
                        bottom: 20
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        font: {
                            size: 12
                        }
                    },
                    title: {
                        display: true,
                        text: 'Count',
                        font: {
                            size: 13,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Defect Type',
                        font: {
                            size: 13,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            layout: {
                padding: {
                    left: 20,
                    right: 20,
                    top: 10,
                    bottom: 10
                }
            }
        }
    });
}

// Generate colors for chart
function generateColors(count) {
    const baseColors = [
        'rgba(37, 99, 235, 0.6)',
        'rgba(139, 92, 246, 0.6)',
        'rgba(239, 68, 68, 0.6)',
        'rgba(16, 185, 129, 0.6)',
        'rgba(245, 158, 11, 0.6)',
        'rgba(14, 165, 233, 0.6)',
        'rgba(236, 72, 153, 0.6)',
        'rgba(132, 204, 22, 0.6)'
    ];
    
    const colors = [];
    for (let i = 0; i < count; i++) {
        colors.push(baseColors[i % baseColors.length]);
    }
    return colors;
}

// Download single image
async function downloadImage(filename, imageData, type) {
    try {
        const response = await fetch(`/download/image/${type}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_data: imageData,
                filename: `${type}_${filename}`
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${type}_${filename}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            alert('Download failed');
        }
    } catch (error) {
        console.error('Download error:', error);
        alert('Download failed: ' + error.message);
    }
}

// Download single CSV
async function downloadSingleCSV(detections, filename) {
    try {
        const detectionsWithImage = detections.map(det => ({
            Image: filename,
            'Defect Type': det.defect_type,
            X1: det.x1,
            Y1: det.y1,
            X2: det.x2,
            Y2: det.y2,
            Confidence: det.confidence
        }));
        
        const response = await fetch('/download/csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                detections: detectionsWithImage,
                filename: `detections_${filename.replace(/\.[^/.]+$/, '')}.csv`
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `detections_${filename.replace(/\.[^/.]+$/, '')}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    } catch (error) {
        console.error('CSV download error:', error);
        alert('CSV download failed');
    }
}

// Download combined CSV
document.getElementById('downloadAllCsv').addEventListener('click', async () => {
    const allDetections = [];
    
    allResults.forEach(result => {
        if (!result.error && result.detections) {
            result.detections.forEach(det => {
                allDetections.push({
                    Image: result.filename,
                    'Defect Type': det.defect_type,
                    X1: det.x1,
                    Y1: det.y1,
                    X2: det.x2,
                    Y2: det.y2,
                    Confidence: det.confidence
                });
            });
        }
    });

    if (allDetections.length === 0) {
        alert('No detections to download');
        return;
    }

    try {
        const response = await fetch('/download/csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                detections: allDetections,
                filename: 'pcb_defects_combined_report.csv'
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'pcb_defects_combined_report.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    } catch (error) {
        console.error('Combined CSV download error:', error);
        alert('Download failed');
    }
});

// Download all annotated images as ZIP
document.getElementById('downloadAllZip').addEventListener('click', async () => {
    if (allResults.length === 0) {
        alert('No images to download');
        return;
    }

    try {
        const response = await fetch('/download/zip', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                results: allResults
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            
            if (blob.size === 0) {
                alert('ZIP file is empty. Please try again.');
                return;
            }
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'all_annotated_images.zip';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            const errorData = await response.json();
            alert(`ZIP download failed: ${errorData.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('ZIP download error:', error);
        alert(`Download failed: ${error.message}`);
    }
});

window.addEventListener('load', () => {
    checkBackendConnection();
    setInterval(checkBackendConnection, 30000);
});

// ============ GRID VIEW FUNCTIONALITY ============

let viewMode = 'expanded';

function addViewToggle() {
    if (document.getElementById('viewToggleSection')) return;
    
    const toggleSection = document.createElement('div');
    toggleSection.className = 'grid-view-toggle';
    toggleSection.id = 'viewToggleSection';
    toggleSection.innerHTML = `
        <button class="view-mode-btn" id="viewModeBtn" onclick="toggleViewMode()">
            🔲 Switch to Grid View
        </button>
    `;
    
    resultsSection.insertBefore(toggleSection, resultsContainer);
}

window.toggleViewMode = function() {
    const btn = document.getElementById('viewModeBtn');
    
    if (viewMode === 'expanded') {
        viewMode = 'grid';
        showGridView();
        btn.innerHTML = '📋 Switch to List View';
    } else {
        viewMode = 'expanded';
        showExpandedView();
        btn.innerHTML = '🔲 Switch to Grid View';
    }
};

function showGridView() {
    resultsContainer.innerHTML = '';
   
    const gridContainer = document.createElement('div');
    gridContainer.className = 'results-grid-view';
    gridContainer.id = 'gridViewContainer';
    
    allResults.forEach((result, index) => {
        const card = createGridCard(result, index);
        gridContainer.appendChild(card);
    });
    
    resultsContainer.appendChild(gridContainer);
}

function updateGridView() {
    let gridContainer = document.getElementById('gridViewContainer');
    
    if (!gridContainer) {
        resultsContainer.innerHTML = '';
        gridContainer = document.createElement('div');
        gridContainer.className = 'results-grid-view';
        gridContainer.id = 'gridViewContainer';
        resultsContainer.appendChild(gridContainer);
    }
    
    gridContainer.innerHTML = '';
    allResults.forEach((result, index) => {
        const card = createGridCard(result, index);
        gridContainer.appendChild(card);
    });
}

function createGridCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-preview-card';
    
    if (result.error) {
        card.innerHTML = `
            <div class="preview-details">
                <div class="preview-filename">❌ ${result.filename}</div>
                <div style="color: var(--danger); font-size: 0.75em;">Error: ${result.error}</div>
            </div>
        `;
        return card;
    }
    
    const defectCount = result.defect_count || 0;
    
    card.innerHTML = `
        <img src="${result.annotated_image}" alt="${result.filename}" class="preview-image">
        <div class="preview-details">
            <div class="preview-filename" title="${result.filename}">${result.filename}</div>
            <div class="preview-stats">
                <span class="defect-count-mini ${defectCount === 0 ? 'zero' : ''}">${defectCount} defects</span>
                <span class="click-hint">👆 Click to view</span>
            </div>
        </div>
    `;
    
    card.addEventListener('click', () => openResultModal(result));
    
    return card;
}

function showExpandedView() {
    resultsContainer.innerHTML = '';
    
    allResults.forEach((result, index) => {
        displaySingleResult(result, index);
    });
}

function openResultModal(result) {
    const modal = document.createElement('div');
    modal.className = 'result-modal';
    modal.id = 'resultModal';
    
    const hasDetections = result.detections && result.detections.length > 0;
    
    let detectionsHTML = '';
    if (hasDetections) {
        detectionsHTML = `
            <table class="detections-table">
                <thead>
                    <tr>
                        <th>Defect Type</th>
                        <th>Confidence</th>
                        <th>X1</th>
                        <th>Y1</th>
                        <th>X2</th>
                        <th>Y2</th>
                    </tr>
                </thead>
                <tbody>
                    ${result.detections.map(det => `
                        <tr>
                            <td><strong>${det.defect_type}</strong></td>
                            <td>${(det.confidence * 100).toFixed(2)}%</td>
                            <td>${det.x1}</td>
                            <td>${det.y1}</td>
                            <td>${det.x2}</td>
                            <td>${det.y2}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } else {
        detectionsHTML = '<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #d1fae5, #a7f3d0); border-radius: 12px; color: #065f46; font-weight: 700; font-size: 1.2em;">🎉 No Defects Found!</div>';
    }
    
    modal.innerHTML = `
        <div class="modal-content-box">
            <div class="modal-header-box">
                <div class="modal-title-box">📊 ${result.filename}</div>
                <button class="modal-close-x" onclick="closeResultModal()">×</button>
            </div>
            <div class="modal-body-box">
                <div class="image-comparison">
                    <div class="image-container">
                        <div class="image-label">Original Image</div>
                        <img src="${result.original_image}" alt="Original">
                    </div>
                    <div class="image-container">
                        <div class="image-label">Detected Defects</div>
                        <img src="${result.annotated_image}" alt="Annotated">
                    </div>
                </div>
                
                <div style="margin: 20px 0;">
                    <h3 style="margin-bottom: 15px; color: var(--dark);">
                        ${hasDetections ? `🔴 Found ${result.defect_count} Defect(s)` : '✅ No Defects'}
                    </h3>
                    ${detectionsHTML}
                </div>
                
                <div class="download-buttons">
                    <button class="btn btn-download" onclick='downloadImage("${result.filename}", "${result.annotated_image}", "annotated")'>
                        💾 Download Annotated
                    </button>
                    <button class="btn btn-download" onclick='downloadImage("${result.filename}", "${result.original_image}", "original")'>
                        📥 Download Original
                    </button>
                    ${hasDetections ? `
                        <button class="btn btn-download" onclick='downloadSingleCSV(${JSON.stringify(result.detections)}, "${result.filename}")'>
                            📄 Download CSV
                        </button>
                    ` : ''}
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeResultModal();
        }
    });
}

window.closeResultModal = function() {
    const modal = document.getElementById('resultModal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = 'auto';
    }
};

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeResultModal();
    }
});