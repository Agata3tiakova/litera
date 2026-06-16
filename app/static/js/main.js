const uploadCard = document.querySelector(".upload-card-dark");
const uploadButton = document.getElementById("uploadButton");
const fileInput = document.getElementById("fileInput");

const uploadStatus = document.getElementById("uploadStatus");
const uploadResult = document.getElementById("uploadResult");
const imagePreview = document.getElementById("imagePreview");
const goToResultBtn = document.getElementById("goToResultBtn");

let lastResultId = null;

if (uploadCard && uploadButton && fileInput && uploadStatus && uploadResult && imagePreview && goToResultBtn) {
    uploadButton.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) handleUpload(fileInput.files[0]);
    });

    uploadCard.addEventListener("dragover", e => {
        e.preventDefault();
        uploadCard.classList.add("drag-over");
    });

    uploadCard.addEventListener("dragleave", () => {
        uploadCard.classList.remove("drag-over");
    });

    uploadCard.addEventListener("drop", e => {
        e.preventDefault();
        uploadCard.classList.remove("drag-over");
        if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
    });

    async function handleUpload(file) {

        imagePreview.src = URL.createObjectURL(file);

        uploadStatus.style.display = "block";
        uploadResult.style.display = "none";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/process", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            uploadStatus.style.display = "none";

            if (data.error) {
                alert(data.error);
                return;
            }

            lastResultId = data.id;
            uploadResult.style.display = "block";

        } catch (err) {
            uploadStatus.style.display = "none";
            alert("Ошибка загрузки");
            console.error(err);
        }
    }

    goToResultBtn.addEventListener("click", () => {
        if (lastResultId) {
            window.location.href = `/results/${lastResultId}`;
        }
    });
}
