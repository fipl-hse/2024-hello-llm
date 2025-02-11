document.addEventListener("DOMContentLoaded", function () {
    const inputField = document.getElementById("queryInput");
    const sendButton = document.getElementById("sendButton");
    const outputField = document.getElementById("output");
    const loadingSpinner = document.getElementById("loadingSpinner");

    sendButton.addEventListener("click", async function () {
        const query = inputField.value.trim();

        if (!query) {
            outputField.textContent = "Please enter a query!";
            outputField.style.color = "red";
            return;
        }

        outputField.textContent = "";
        outputField.style.color = "black";
        loadingSpinner.style.display = "block";

        try {
            const response = await fetch("/infer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: query })
            });

            const result = await response.json();
            outputField.textContent = `AI says: ${result.infer}`;
        } catch (error) {
            outputField.textContent = "Error fetching response!";
            outputField.style.color = "red";
        } finally {
            loadingSpinner.style.display = "none";
        }
    });
});
