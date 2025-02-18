const translateButton = document.getElementById("translate-button");
const handleTranslationClick = async () => {
    try {
        const russianText = document.getElementById("source").value;
        const translation = document.getElementById("target");

        if (!russianText) {
            translation.textContent = "Введите текст!";
            return;
        }

        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({question: russianText})
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        translation.innerText = data.infer;

    } catch (error) {
        console.error('Error during translation:', error);
        translation.textContent = `Error: ${error.message}`;
    }
};

translateButton.addEventListener('click', handleTranslationClick);
