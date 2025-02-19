const button = document.getElementById('buttonInput');

button.addEventListener('click', async () => {

    const textInput = document.getElementById('textInput').value;
    const resultDiv = document.getElementById('result');

    let response = await fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: textInput })
    });

    if (response.ok) {
        let result = await response.json();
        resultDiv.textContent = result.infer;
    } else {
        alert('Ошибка HTTP: ' + response.status);
        resultDiv.textContent = `Ошибка HTTP: ${await response.status}`;
    }

});
