const checkbox = document.getElementById('base-model-checkbox');
const clickButton = document.getElementById("send-button");
const handleClick = async () => {
    try {
        const premise = document.getElementById("premise").value;
        const hypothesis = document.getElementById("hypothesis").value;
        const isChecked = checkbox.checked;
        const result = document.getElementById("prediction");

        if (!premise || !hypothesis) {
            result.textContent = "You need to fill both text areas!";
            return;
        }

        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: premise,
                hypothesis: hypothesis,
                is_base_model: isChecked})
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        result.innerText = data.infer;

    } catch (error) {
        console.error('Error during inference:', error);
        result.textContent = `Error: ${error.message}`;
    }
};

clickButton.addEventListener('click', handleClick);
