document.addEventListener("DOMContentLoaded", function () {
    const button = document.getElementById("submit-btn");
    const inputText = document.getElementById("input-text");
    const responseParagraph = document.getElementById("response");

    button.addEventListener("click", async function () {
        const text = inputText.value.trim();

        if (!text) {
            responseParagraph.innerText = "Введите текст!";
            return;
        }

        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: text })
        });

        const data = await response.json();
        responseParagraph.innerText = data.infer;
    });
});
