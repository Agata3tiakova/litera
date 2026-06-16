const textarea = document.getElementById("recognizedText");

const symbolsCount = document.getElementById("symbolsCount");
const wordsCount = document.getElementById("wordsCount");
const sentencesCount = document.getElementById("sentencesCount");

const copyBtn = document.getElementById("copyTextBtn");
const clearBtn = document.getElementById("clearTextBtn");
const analyzeBtn = document.getElementById("startAnalysisBtn");

const progressBlock = document.getElementById("analysisProgress");
const progressBar = progressBlock ? progressBlock.querySelector(".progress-bar") : null;
const progressText = document.getElementById("progressText");

if (textarea && symbolsCount && wordsCount && sentencesCount && copyBtn && clearBtn && analyzeBtn && progressBlock && progressBar && progressText) {
    function updateStats() {
        const text = textarea.value;

        symbolsCount.textContent = text.length;
        wordsCount.textContent = text.trim()
            ? text.trim().split(/\s+/).length
            : 0;
        sentencesCount.textContent = text.split(/[.!?]+/).filter(Boolean).length;
    }

    textarea.addEventListener("input", updateStats);
    document.addEventListener("DOMContentLoaded", updateStats);

    copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(textarea.value);
    });

    clearBtn.addEventListener("click", () => {
        textarea.value = "";
        updateStats();
    });

    analyzeBtn.addEventListener("click", async () => {
        const text = textarea.value.trim();

        if (!text) {
            alert("Текст пуст");
            return;
        }

        progressBlock.style.display = "block";
        progressBar.style.width = "30%";
        progressText.textContent = "Отправка текста…";

        try {
            const response = await fetch(`/api/analyze/${RESULT_ID}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });

            progressBar.style.width = "80%";
            progressText.textContent = "AI анализирует текст…";

            if (!response.ok) {
                throw new Error("Ошибка анализа");
            }

            progressBar.style.width = "100%";
            progressText.textContent = "Готово!";

            window.location.href = `/analysis/${RESULT_ID}`;

        } catch (err) {
            alert("Ошибка анализа");
            console.error(err);
        }
    });
}
