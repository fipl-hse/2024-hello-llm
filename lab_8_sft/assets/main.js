const textInput = document.querySelector('#user-full-text-input');
const submitButton = document.querySelector('#detect-btn');
const resultDisplay = document.querySelector('#response');
const modelCheckbox = document.querySelector('#base-model-checkbox');

submitButton.addEventListener('click', async () => {
    const inputText = textInput.value.trim();
    if (!inputText) {
        resultDisplay.textContent = "Please enter some text!";
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

        let { infer } = await response.json();
        infer = infer === "0" ? "This is not toxic content!" : "This is toxic content!";
        resultDisplay.textContent = infer;
    } catch (error) {
        resultDisplay.textContent = `Error: ${error.message}`;
    }
});
