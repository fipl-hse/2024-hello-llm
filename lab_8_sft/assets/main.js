const classifyButton = document.getElementById('classifyButton');

classifyButton.addEventListener('click', async () => {
    const inputText = document.getElementById('inputText').value;
    const useBaseModel = document.getElementById('useBaseModel').checked;
    const classificationResult = document.getElementById('result');

    if (!inputText.trim()) {
        classificationResult.textContent = 'Enter some text you\'d like to classify!';
        return;
    }

    classificationResult.textContent = 'Classifying..';

    try {
        const response = await fetch('/infer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: inputText, use_base_model: useBaseModel }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        classificationResult.textContent = data.infer;

    } catch (error) {
        classificationResult.textContent = 'An error occurred during text classification :(' + error.message;
        console.error(`Error: ${error.message}`);
    }
});
