const FILES_STATE = JSON.parse(sessionStorage.getItem("FILES_STATE") || "[]");
const idx = sessionStorage.getItem("SELECTED_INDEX");
const item = FILES_STATE[idx];

if (!item || item.status === "NOT_PCB") {
  alert("This file is not a PCB image.");
  window.location.href = "/";
}

document.getElementById("fileName").innerText = item.file;
document.getElementById("origImg").src = item.image_path;
document.getElementById("annotImg").src = item.annotated_path;
document.getElementById("conf").innerText = item.pcb_confidence.toFixed(3);
document.getElementById("defects").innerText = item.defect_count;
document.getElementById("time").innerText = item.inference_ms;

document.getElementById("pdfLink").href = item.pdf_path;
document.getElementById("jsonLink").href = item.json_path;

/* Charts */
Plotly.newPlot("barChart", [{
  x: item.defects.map((d, i) => `${d.label} #${i + 1}`),
  y: item.defects.map(d => d.confidence),
  type: "bar"
}], { title: "Confidence per Defect" });

const counts = {};
item.defects.forEach(d => counts[d.label] = (counts[d.label] || 0) + 1);

Plotly.newPlot("pieChart", [{
  labels: Object.keys(counts),
  values: Object.values(counts),
  type: "pie"
}], { title: "Defect Type Distribution" });

const tbody = document.getElementById("defectTable");
tbody.innerHTML = "";
item.defects.forEach((d, i) => {
  tbody.innerHTML += `
    <tr>
      <td>${i + 1}</td>
      <td>${d.label}</td>
      <td>${d.confidence.toFixed(3)}</td>
      <td>${JSON.stringify(d.bbox)}</td>
    </tr>`;
});
