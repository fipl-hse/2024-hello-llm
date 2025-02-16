const textInput = document.querySelector('#user-full-text-input');
const submitButton = document.querySelector('#summarize-btn');
const resultDisplay = document.querySelector('#response');

submitButton.addEventListener('click', async () => {
    const inputText = textInput.value.trim();
    if (!inputText) return resultDisplay.textContent = "No text provided!";

    resultDisplay.textContent = "Handling your request... please hold on!";

    try {
        const response = await fetch("/infer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: inputText })
        });

        if (!response.ok) throw new Error(`Server error: ${await response.text()}`);

        const { infer } = await response.json();
        resultDisplay.textContent = infer;
    } catch (error) {
        resultDisplay.textContent = `Error: ${error.message}`;
    }
});