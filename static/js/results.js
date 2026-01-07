// Results carousel functionality
let currentIndex = 0;
let resultsData = [];

document.addEventListener('DOMContentLoaded', function () {
    // Access results from the global window object safely
    if (window.results && Array.isArray(window.results)) {
        resultsData = window.results;
        console.log("Loaded results data:", resultsData.length, "items");
        if (resultsData.length > 0) {
            displayResult(0);
        } else {
            console.warn("Results array is empty");
        }
    } else {
        console.error("No results data found on window object");
    }

    setupCarouselListeners();
    setupDownloadListeners();
});

function setupCarouselListeners() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    if (prevBtn) {
        prevBtn.addEventListener('click', () => navigateCarousel(-1));
    } else {
        console.error("Previous button not found");
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => navigateCarousel(1));
    } else {
        console.error("Next button not found");
    }
}

function setupDownloadListeners() {
    const downloadImgBtn = document.getElementById('download-img-btn');
    const downloadTxtBtn = document.getElementById('download-txt-btn');
    const downloadZipBtn = document.getElementById('download-zip-btn');

    if (downloadImgBtn) {
        downloadImgBtn.addEventListener('click', downloadCurrentImage);
    }

    if (downloadTxtBtn) {
        downloadTxtBtn.addEventListener('click', downloadCurrentAnnotation);
    }

    if (downloadZipBtn) {
        downloadZipBtn.addEventListener('click', downloadZip);
    }
}

function navigateCarousel(direction) {
    if (resultsData.length === 0) return;

    currentIndex += direction;

    if (currentIndex < 0) {
        currentIndex = resultsData.length - 1;
    } else if (currentIndex >= resultsData.length) {
        currentIndex = 0;
    }

    displayResult(currentIndex);
}

function displayResult(index) {
    if (index < 0 || index >= resultsData.length) return;

    const result = resultsData[index];
    currentIndex = index;

    console.log("Displaying result:", index, result);

    // Update header
    const nameEl = document.getElementById('current-image-name');
    const counterEl = document.getElementById('carousel-counter');

    if (nameEl) nameEl.textContent = `📷 ${result.original_name}`;
    if (counterEl) counterEl.textContent = `${index + 1} / ${resultsData.length}`;

    // Update images with timestamp to prevent caching issues
    const originalImg = document.getElementById('original-image');
    const annotatedImg = document.getElementById('annotated-image');
    const timestamp = new Date().getTime();

    if (originalImg && result.original_filename) {
        originalImg.src = `/results/${result.original_filename}?t=${timestamp}`;
        originalImg.alt = result.original_name;
    }

    if (annotatedImg && result.annotated_filename) {
        annotatedImg.src = `/results/${result.annotated_filename}?t=${timestamp}`;
        annotatedImg.alt = result.annotated_name;
    }

    // Update annotations
    const annotationsContent = document.getElementById('annotations-content');
    if (annotationsContent) {
        if (result.annotation_content && result.annotation_content.trim()) {
            const lines = result.annotation_content.trim().split('\n').filter(line => line.trim());
            const prettyLines = lines.map((line, idx) => {
                const parts = line.split(/\s+/);
                if (parts.length === 5) {
                    const [clsId, cx, cy, w, h] = parts;
                    const className = getClassName(parseInt(clsId));
                    return `#${idx + 1}  class=${clsId} (${className})  cx=${cx}, cy=${cy}, w=${w}, h=${h}`;
                }
                return line;
            });
            annotationsContent.textContent = prettyLines.join('\n');
        } else {
            annotationsContent.textContent = 'No detections (empty annotation file)';
        }
    }

    // Update download buttons data attributes
    const downImgBtn = document.getElementById('download-img-btn');
    const downTxtBtn = document.getElementById('download-txt-btn');

    if (downImgBtn) {
        downImgBtn.dataset.filename = result.annotated_filename; // Use correct property
    }
    if (downTxtBtn) {
        downTxtBtn.dataset.filename = result.annotation_filename; // Use correct property
    }

    // Update navigation buttons state
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    if (prevBtn) prevBtn.disabled = resultsData.length <= 1;
    if (nextBtn) nextBtn.disabled = resultsData.length <= 1;
}

function getClassName(classId) {
    const classNames = {
        0: "Missing_hole",
        1: "Mouse_bite",
        2: "Open_circuit",
        3: "Short",
        4: "Spur",
        5: "Spurious_copper"
    };
    return classNames[classId] || "unknown";
}

function downloadCurrentImage() {
    const btn = document.getElementById('download-img-btn');
    const filename = btn.dataset.filename;
    if (filename) {
        window.location.href = `/api/download_image/${filename}`;
    } else {
        console.error("No filename found on download button");
    }
}

function downloadCurrentAnnotation() {
    const btn = document.getElementById('download-txt-btn');
    const filename = btn.dataset.filename;
    if (filename) {
        window.location.href = `/api/download_annotation/${filename}`;
    } else {
        console.error("No filename found on download button");
    }
}

function downloadZip() {
    window.location.href = '/api/download_zip';
}

