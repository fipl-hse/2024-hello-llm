const summarizeButton = document.getElementById('summarizeButton');

summarizeButton.addEventListener('click', async () => {
    const inputText = document.getElementById('inputText').value;
    const summaryResult = document.getElementById('result');

    if (!inputText.trim()) {
        summaryResult.textContent = 'Please enter some text to summarize.';
        return;
    }

    summaryResult.textContent = 'Summarizing...';

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
        summaryResult.textContent = data.infer || "We couldn't summarize your text :(";

    } catch (error) {
        summaryResult.textContent = 'An error occurred while summarizing the text.';
        console.error('Error:', error);
    }
});
