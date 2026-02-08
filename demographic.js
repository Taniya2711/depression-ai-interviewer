const input = document.getElementById("cvInput");
const uploadText = document.getElementById("uploadText");
const progressBar = document.getElementById("progressBar");
const progressContainer = document.getElementById("progressContainer");
const uploadContent = document.getElementById("uploadContent");

function triggerUpload() {
    input.click();
}

input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;

    progressContainer.classList.remove("hidden");
    progressBar.style.width = "0%";
    uploadText.textContent = "Uploading...";

    let progress = 0;
    const interval = setInterval(() => {
        progress += 12;
        progressBar.style.width = progress + "%";

        if (progress >= 100) {
            clearInterval(interval);
            progressContainer.classList.add("hidden");

            uploadContent.innerHTML = `
                <div style="font-size:18px;">📄 ${file.name}</div>
                <div style="font-size:13px;color:#94a3b8;margin-top:6px;">
                    Upload complete
                </div>
            `;
        }
    }, 120);
});

function continueInterview() {
    window.location.href = "index.html";
}
