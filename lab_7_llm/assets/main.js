document.getElementById('summarizeButton').addEventListener('click', async () => {
    const inputText = document.getElementById('inputText').value;
    const resultDiv = document.getElementById('result');

    if (!inputText) {
        resultDiv.textContent = 'Please enter some text to summarize.';
        return;
    }

    try {
        const response = await fetch('/infer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: inputText }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        resultDiv.textContent = data.infer;
    } catch (error) {
        resultDiv.textContent = 'An error occurred while summarizing the text.';
        console.error('Error:', error);
    }
});