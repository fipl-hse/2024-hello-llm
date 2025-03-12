function Summarize() {
    const useBasemodel = document.getElementById('is_base_model').checked;
    const inputText = document.getElementById('input-text').value;

    if (!inputText.trim()) {
        alert("Please enter some text for summarization");
        return;
    }

    fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            is_base_model: useBasemodel,
            question: inputText
        })
    })
    .then(response => response.json())
    .then(data => {
        const resultElement = document.getElementById('result');
        resultElement.innerText = `Summarization: ${data.infer}.`;
        resultElement.classList.add('active');
    })
    .catch(error => {
        console.error('Error:', error);
        alert("An error occurred while performing summarization.");
    });
}
