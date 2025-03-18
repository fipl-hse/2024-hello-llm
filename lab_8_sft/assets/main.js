const button = document.getElementById('buttonInput');
const checkbox = document.getElementById('checkboxInput')

button.addEventListener('click', async () => {

    const textInput = document.getElementById('textInput').value;
    const resultDiv = document.getElementById('result');

    if (checkbox.checked) {
        useBaseModel = true
    } else {
        useBaseModel = false
    };

    let response = await fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({question: textInput, is_base_model: useBaseModel})
    });

    if (response.ok) {
        let result = await response.json();
        resultDiv.textContent = result.infer;
    } else {
        alert('Ошибка HTTP: ' + response.status);
        resultDiv.textContent = `Ошибка HTTP: ${await response.status}`;
    }

});
