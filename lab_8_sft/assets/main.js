const userInput = document.getElementById('input');
const button = document.getElementById('button');
const response = document.getElementById('response');
const checkbox = document.getElementById('checkbox')

function updateResponse(text) {
  response.textContent = text;
}

button.addEventListener('click', async () => {
    const userText = userInput.value.trim();

    updateResponse('Just a second...');

    const isBaseModel = checkbox.checked;

    try {
        let response = await fetch("/infer", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ question: userText, is_base_model: isBaseModel })
        });

        if (response.ok) {
            let data = await response.json();
            updateResponse(data.infer);
        } else {
            alert(`HTTP-error: ${response.status}`);
        }
    }
    catch (error) {
        updateResponse(`Error: ${error.message}`);
    }
});
