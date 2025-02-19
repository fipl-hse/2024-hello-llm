const summarizeButton = document.getElementById('summarizeButton');

summarizeButton.addEventListener('click', async () => {
    const inputText = document.getElementById('inputText').value;
    const summaryResult = document.getElementById('result');

    if (!inputText.trim()) {
        summaryResult.textContent = 'Please enter some text to summarize';
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

        const trimmedInfer = (data.infer || '').trim();

        if (trimmedInfer === "") {
            summaryResult.textContent = 'We could not summarize your text :('; //
        } else {
            summaryResult.textContent = trimmedInfer; //
        }

    } catch (error) {
        summaryResult.textContent = 'An error occurred during text summarization :(';
        console.error(`Error: ${error.message}`);
    }
});
