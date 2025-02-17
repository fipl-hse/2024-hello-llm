const button = document.getElementById('buttonInput');
console.log('button created');

button.addEventListener('click', async () => {

    const textInput = document.getElementById('textInput');
    const resultDiv = document.getElementById('result');

    resultDiv.innerText = 'Processing...';
    const text = textInput.value.trim();

let response = await fetch('/infer', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    }
    body: JSON.stringify({ question: text })
});

if (response.ok) {
    let result = await response.json();
} else {
    alert('Ошибка HTTP: ' + response.status);
}

resultDiv.innerText = result.infer;
console.log('result sent to server');
});
