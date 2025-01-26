const userInput = document.getElementById('user-input');
const submitBtn = document.getElementById('submit-btn');
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

    try {
        const response = await fetch("/summarize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: userText })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        updateResponseText(data.infer);
    } catch (error) {
        updateResponseText('Error: ' + error.message);
    }
});