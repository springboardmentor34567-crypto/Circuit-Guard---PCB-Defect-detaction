document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("processBtn");
  const overlay = document.getElementById("loadingOverlay");
  const zipLink = document.getElementById("zipLink");

  let FILES_STATE = JSON.parse(
    sessionStorage.getItem("FILES_STATE") || "[]"
  );

  if (performance.navigation.type === 1) {
    sessionStorage.clear();
    FILES_STATE = [];
  }

  if (FILES_STATE.length) renderTable();

  btn.addEventListener("click", processImages);

  async function processImages() {
    const files = document.getElementById("fileInput").files;
    if (!files.length) return alert("Select images");

    overlay.classList.remove("hidden");

    try {
      const fd = new FormData();
      [...files].forEach(f => fd.append("files", f));

      const res = await fetch("/process", {
        method: "POST",
        body: fd
      });

      const data = await res.json();

      FILES_STATE = data.results;
      sessionStorage.setItem(
        "FILES_STATE",
        JSON.stringify(FILES_STATE)
      );

      if (data.zip_url) {
        zipLink.href = data.zip_url;
        zipLink.classList.remove("hidden");
      } else {
        zipLink.classList.add("hidden");
      }

      renderTable();
    } finally {
      overlay.classList.add("hidden");
    }
  }

  function renderTable() {
    const tbody = document.getElementById("resultsTable");
    tbody.innerHTML = "";

    let sum = 0, cnt = 0, defects = 0;

    FILES_STATE.forEach((r, i) => {
      if (r.status !== "NOT_PCB") {
        sum += r.pcb_confidence;
        cnt++;
        defects += r.defect_count;
      }

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.file}</td>
        <td>${r.pcb_confidence.toFixed(3)}</td>
        <td>${r.defect_count}</td>
        <td>${r.status}</td>
        <td>
          ${r.status === "NOT_PCB"
            ? "-"
            : `<button onclick="openDetails(${i})">View</button>`}
        </td>
      `;
      tbody.appendChild(tr);
    });

    document.getElementById("totalFiles").innerText = FILES_STATE.length;
    document.getElementById("avgConf").innerText =
      cnt ? (sum / cnt).toFixed(3) : "-";
    document.getElementById("totalDefects").innerText = defects;
  }

  window.openDetails = function (idx) {
    sessionStorage.setItem("SELECTED_INDEX", idx);
    window.location.href = "/frontend/details.html";
  };
});
