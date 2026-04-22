async function pollJob(jobId) {
  const statusEl = document.querySelector("[data-job-status]");
  const detailEl = document.querySelector("[data-job-detail]");
  const artifactsEl = document.querySelector("[data-job-artifacts]");

  if (!jobId || !statusEl) {
    return;
  }

  async function refresh() {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) {
      return;
    }
    const job = await response.json();
    statusEl.textContent = job.status;
    detailEl.textContent = job.message || "";

    if (job.status === "done" || job.status === "error") {
      if (artifactsEl) {
        artifactsEl.innerHTML = job.artifacts
          .map(
            (artifact) => `
              <div class="artifact">
                <div>
                  <strong>${artifact.name}</strong>
                  <div><span>${artifact.filename}</span></div>
                </div>
                <a class="btn secondary" href="${artifact.download_url}">Download</a>
              </div>
            `
          )
          .join("");
      }
      return;
    }

    window.setTimeout(refresh, 2000);
  }

  refresh();
}

window.addEventListener("DOMContentLoaded", () => {
  const jobId = document.body.getAttribute("data-job-id");
  pollJob(jobId);
});

