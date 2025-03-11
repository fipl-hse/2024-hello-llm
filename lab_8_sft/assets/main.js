document.addEventListener('DOMContentLoaded', function () {
    const submitButton = document.getElementById('submitButton');
    const textInput = document.getElementById('textInput');
    const responseField = document.getElementById('responseField');
    const baseModelCheckbox = document.getElementById('base-model-checkbox');

    submitButton.addEventListener('click', async function (event) {
        event.preventDefault();
        submitButton.classList.add('clicked')
        const inputText = textInput.value.trim();

        if (!inputText) {
            responseField.innerText = 'You forgot to enter the text!';
            return;
        }

        const useBaseModel = baseModelCheckbox.checked;

        const requestData = {
            question: inputText,
            use_base_model: useBaseModel
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

            responseField.innerText = `This text is classified as: ${responseData.infer}`;
        } catch (error) {
            responseField.innerText = `Error: ${error.message}`;
        }
    });
});