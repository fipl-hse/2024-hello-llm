document.addEventListener("DOMContentLoaded", function () {
    const button = document.getElementById("send-btn");
    const input = document.getElementById("question");
    const responseElement = document.getElementById("response");

    button.addEventListener("click", async function () {
        const questionText = input.value;
        if (!questionText.trim()) {
            responseElement.textContent = "Where is your text?";
            return;
        }

        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: questionText })
        });

        const result = await response.json();
        responseElement.textContent = result.infer;
    });
});