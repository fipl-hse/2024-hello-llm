const userInput = document.getElementById('input');
const submitBtn = document.getElementById('button');
const responseText = document.getElementById('response');

function updateResponseText(text) {
  responseText.textContent = text;
}

submitBtn.addEventListener('click', async () => {
    const userText = userInput.value.trim();

    updateResponseText('Just a second...');

    try {
        let response = await fetch("/infer", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ question: userText })
        });

        if (response.ok) {
            let data = await response.json();
            updateResponseText(data.infer);
        } else {
            alert("HTTP-error: " + response.status);
        }
    }
    catch (error) {
        updateResponseText(`Error: ${error.message}`);
    }
});