document.getElementById("queryForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const inputText = document.getElementById("inputText").value;

    try {
        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: inputText })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        document.getElementById("response").innerText = `Response: ${data.infer}`;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("response").innerText = "An error occurred.";
    }
});
