document.addEventListener('DOMContentLoaded', function () {
    const submitButton = document.getElementById('submitButton');
    const textInput = document.getElementById('textInput');
    const responseField = document.getElementById('responseField');
    const baseModelCheckbox = document.getElementById('base-model-checkbox');

    submitButton.addEventListener('click', async function (event) {
        event.preventDefault();

        const inputText = textInput.value.trim();

        if (!inputText) {
            responseField.innerText = 'You forgot to enter the text!';
            return;
        }

        const isBaseModel = baseModelCheckbox.checked;

        const requestData = {
            question: inputText,
            is_base_model: isBaseModel
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
            responseField.innerText = `Error: ${error.message}`;
        }
    });
});
