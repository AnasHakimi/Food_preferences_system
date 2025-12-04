document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultContainer = document.getElementById('result');
    const predictionText = document.getElementById('predictionText');
    const resetBtn = document.getElementById('resetBtn');
    const predictBtn = document.getElementById('predictBtn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        const originalBtnText = predictBtn.innerText;
        predictBtn.innerText = 'Analyzing...';
        predictBtn.disabled = true;

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                predictionText.innerText = result.prediction;
                form.classList.add('hidden');
                resultContainer.classList.remove('hidden');
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            alert('An error occurred. Please try again.');
            console.error(error);
        } finally {
            predictBtn.innerText = originalBtnText;
            predictBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        form.classList.remove('hidden');
        form.reset();
        // Reset outputs for sliders
        document.querySelectorAll('output').forEach(o => o.innerText = '5');
    });
});
