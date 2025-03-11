function Summarize() {
    const useFinetuned = document.getElementById('use_finetuned').checked;
    const inputText = document.getElementById('input-text').value;

    if (!inputText.trim()) {
        alert("Please enter some text for summarization");
        return;
    }

    fetch('/inference', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            use_finetuned: useFinetuned,
            text: inputText
        })
    })
    .then(response => response.json())
    .then(data => {
        const resultElement = document.getElementById('result');
        resultElement.innerText = `Summarization: ${data.prediction}.`;
        resultElement.classList.add('active');
    })
    .catch(error => {
        console.error('Error:', error);
        alert("An error occurred while performing summarization.");
    });
}
