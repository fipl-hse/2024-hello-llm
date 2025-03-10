const userInput = document.getElementById('user-full-text-input');
const submitBtn = document.getElementById('summarize-btn');
const responseText = document.getElementById('response');

const updateResponseText = (message) => {
    responseText.textContent = message;
};

submitBtn.addEventListener('click', async () => {
    const userText = userInput.value.trim();

    if (userText === '') {
        updateResponseText('Please enter some text');
        return;
    }

    updateResponseText('Waiting for the answer...')

    try {
        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: userText })
        });

        if (!response.ok) {
            const errorMessage = await response.text()
            throw new Error(`Network response was not ok: ${errorMessage}`);
        }

        const data = await response.json();
        updateResponseText(data.infer);
    } catch (error) {
        updateResponseText(`Error!  ${error.message}`);
    }
});
