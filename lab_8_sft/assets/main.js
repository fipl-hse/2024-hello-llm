document.addEventListener("DOMContentLoaded", function () {
    const button = document.getElementById("submit-btn");
    const inputText = document.getElementById("input-text");
    const responseParagraph = document.getElementById("response");
    const checkbox = document.getElementById("use-base-model");

    button.addEventListener("click", async function () {
        const text = inputText.value.trim();
        const isBaseModel = checkbox.checked;

        if (!text) {
            responseParagraph.innerText = "Type!";
            return;
        }

        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: text, is_base_model: isBaseModel })
        });

        const data = await response.json();
        responseParagraph.innerText = data.infer;
    });
});
