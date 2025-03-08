document.addEventListener('DOMContentLoaded', function () {
    const submitButton = document.getElementById('submitButton');
    const textInput = document.getElementById('textInput');
    const responseField = document.getElementById('responseField');

    submitButton.addEventListener('click', async function (event) {
        event.preventDefault();
        const inputText = textInput.value.trim();

        if (!inputText) {
            responseField.innerText = 'You forgot to enter the text!';
            return;
        }

        const requestData = {
            question: inputText
        };

        try {
            const response = await fetch('/infer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error('An error occurred while sending the request');
            }

            const responseData = await response.json();
            responseField.innerText = `The answer: ${responseData.infer}`;
        } catch (error) {
            responseField.innerText = Error: ${error.message}`;
        }
    });
});
