const textInput = document.querySelector('#user-full-text-input');
const submitButton = document.querySelector('#detect-btn');
const resultDisplay = document.querySelector('#response');
const modelCheckbox = document.querySelector('#base-model-checkbox');

submitButton.addEventListener('click', async () => {
    const inputText = textInput.value.trim();
    if (!inputText) {
        resultDisplay.textContent = "Enter the text here...";
        return;
    }

    resultDisplay.textContent = "Processing new query... please wait!";
    const isBaseModel = modelCheckbox.checked;

    try {
        const response = await fetch("/infer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: inputText, is_base_model: isBaseModel })
        });

        if (!response.ok) throw new Error(`Server error: ${await response.text()}`);

        const { infer } = await response.json();
        resultDisplay.textContent = infer;
    } catch (error) {
        resultDisplay.textContent = `Error: ${error.message}`;
    }
});
