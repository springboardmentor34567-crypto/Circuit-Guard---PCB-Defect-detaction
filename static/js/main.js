document.addEventListener('DOMContentLoaded', function() {
    // --- Elements ---
    const runBtn = document.getElementById('runBtn');
    const fileInput = document.getElementById('fileInput');
    const confLow = document.getElementById('confLow');
    const confHigh = document.getElementById('confHigh');
    const resultsGrid = document.getElementById('resultsGrid');
    const welcomeScreen = document.getElementById('welcomeScreen');
    
    // Progress Bar Elements
    const progressSection = document.getElementById('progressSection');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressStatus = document.getElementById('progressStatus');

    // Stats Elements
    const statTotal = document.getElementById('statTotal');
    const statDefects = document.getElementById('statDefects');

    // --- Event Listeners ---

    // 1. Enable Run Button only when files are selected
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            runBtn.disabled = false;
            runBtn.innerHTML = `🚀 Run Prediction (${this.files.length} Files)`;
        } else {
            runBtn.disabled = true;
            runBtn.innerHTML = `🚀 Run Prediction`;
        }
    });

    // 2. Update Slider Labels Live
    if (confLow) {
        confLow.addEventListener('input', function() {
            document.getElementById('confLowVal').textContent = this.value;
        });
    }
    if (confHigh) {
        confHigh.addEventListener('input', function() {
            document.getElementById('confHighVal').textContent = this.value;
        });
    }

    // 3. MAIN ACTION: Click "Run Prediction"
    runBtn.addEventListener('click', async function() {
        if (fileInput.files.length === 0) return;

        // A. Setup UI for Loading
        runBtn.disabled = true;
        welcomeScreen.classList.add('d-none');
        resultsGrid.innerHTML = ''; 
        
        // Show Progress Bar
        progressSection.classList.remove('d-none');
        progressBar.style.width = '0%';
        progressBar.classList.remove('bg-success', 'bg-danger');
        progressBar.classList.add('bg-info');
        progressStatus.textContent = "Uploading & Analyzing...";

        // B. Fake Progress Animation (Visual Only)
        let progress = 0;
        const interval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 5) + 1;
                if (progress > 90) progress = 90;
                progressBar.style.width = `${progress}%`;
                progressPercent.textContent = `${progress}%`;
            }
        }, 100);

        // C. Prepare Data
        const formData = new FormData();
        
        // Correctly append files with 'files[]'
        for (const file of fileInput.files) {
            formData.append('files[]', file);
        }
        
        if(confLow) formData.append('conf_threshold', confLow.value); 
        if(confHigh) formData.append('iou_threshold', confHigh.value); 

        try {
            // D. Send to Backend
            const response = await fetch('/inspection/predict', { 
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(errText || "Server Error");
            }

            // E. Handle Response
            const data = await response.json(); 
            
            // --- FIX IS HERE ---
            // Extract list from the object
            let resultsList = [];
            if (Array.isArray(data)) {
                resultsList = data;
            } else if (data.results && Array.isArray(data.results)) {
                resultsList = data.results;
            } else {
                console.warn("Unexpected data format:", data);
            }

            const totalCount = resultsList.length;
            const defectCount = resultsList.filter(r => r.status === 'Defect').length;
            // -------------------

            // F. Finish Animation
            clearInterval(interval);
            progressBar.style.width = '100%';
            progressPercent.textContent = '100%';
            progressBar.classList.remove('bg-info');
            progressBar.classList.add('bg-success');
            progressStatus.textContent = "Analysis Complete!";

            // G. Render Results
            renderResults(resultsList); // Pass the list, not the raw data
            updateStats(totalCount, defectCount);

            // Hide bar after 1.5 seconds
            setTimeout(() => {
                progressSection.classList.add('d-none');
                runBtn.disabled = false;
            }, 1500);

        } catch (error) {
            console.error(error);
            clearInterval(interval);
            progressBar.classList.remove('bg-info');
            progressBar.classList.add('bg-danger');
            progressStatus.textContent = "Error: " + error.message;
            runBtn.disabled = false;
        }
    });

    // --- Helper Functions ---

    function renderResults(results) {
        if (!results || results.length === 0) {
            resultsGrid.innerHTML = '<div class="text-white">No results found.</div>';
            return;
        }

        results.forEach(item => {
            const col = document.createElement('div');
            col.className = 'col-md-4 col-lg-3 mb-4';
            
            const badgeClass = item.status === 'Pass' ? 'bg-success' : 'bg-danger';
            const borderClass = item.status === 'Pass' ? 'border-success' : 'border-danger';
            
            let defectsHtml = '';
            if (item.defects && item.defects.length > 0) {
                item.defects.forEach(d => {
                    defectsHtml += `<span class="badge bg-secondary me-1 mb-1">${d.label} ${(d.confidence*100).toFixed(0)}%</span>`;
                });
            } else {
                defectsHtml = '<small class="text-muted">No defects detected</small>';
            }

            const imgSrc = item.thumbnail ? `data:image/jpeg;base64,${item.thumbnail}` : `/static/uploads/${item.filename}`;

            col.innerHTML = `
                <div class="card h-100 bg-dark text-white shadow-sm defect-card ${borderClass}" style="border-top: 5px solid;">
                    <img src="${imgSrc}" class="card-img-top img-thumbnail-grid" alt="${item.filename}" style="height: 200px; object-fit: cover;">
                    <div class="card-body p-2">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="card-title text-truncate mb-0" title="${item.filename}">${item.filename}</h6>
                            <span class="badge ${badgeClass}">${item.status}</span>
                        </div>
                        <div class="mb-1">
                            ${defectsHtml}
                        </div>
                    </div>
                </div>
            `;
            resultsGrid.appendChild(col);
        });
    }

    function updateStats(total, defects) {
        if(statTotal) statTotal.innerText = total;
        if(statDefects) statDefects.innerText = defects;
    }
});