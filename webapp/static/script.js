document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const detectBtn = document.getElementById('detectBtn');
    const resetBtn = document.getElementById('resetBtn');
    
    const inputSection = document.querySelector('.input-section');
    const loadingSection = document.getElementById('loadingSection');
    const resultSection = document.getElementById('resultSection');

    // Example buttons
    window.setExample = (text) => {
        textInput.value = text;
        textInput.focus();
    };

    detectBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert("Please enter some text!");
            return;
        }

        // Show loading state
        inputSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error("Prediction Error:", error);
            alert("An error occurred during prediction. Please make sure the backend is running.");
            resetUI();
        }
    });

    resetBtn.addEventListener('click', () => {
        resetUI();
        textInput.value = "";
    });

    function renderResults(data) {
        // Hide loader
        loadingSection.classList.add('hidden');
        
        // Setup prediction text
        const predText = document.getElementById('predictionText');
        if (data.sarcastic) {
            predText.innerText = "SARCASTIC";
            predText.className = "is-sarcastic";
        } else {
            predText.innerText = "NOT SARCASTIC";
            predText.className = "not-sarcastic";
        }

        // Output confidence and emotion
        const confidencePercentage = (data.confidence * 100).toFixed(1);
        document.getElementById('confidenceValue').innerText = `${confidencePercentage}%`;
        document.getElementById('emotionValue').innerText = data.emotion;

        // Render trajectory
        const flowContainer = document.getElementById('trajectoryFlow');
        flowContainer.innerHTML = ""; // Clear existing

        data.trajectory.forEach((emotion, index) => {
            // Delay rendering for slick animation
            setTimeout(() => {
                const node = document.createElement('div');
                node.className = 'traj-node';
                node.innerText = emotion;
                flowContainer.appendChild(node);

                // Add arrow if not the last node
                if (index < data.trajectory.length - 1) {
                    const arrow = document.createElement('div');
                    arrow.className = 'traj-arrow';
                    arrow.innerText = '→';
                    flowContainer.appendChild(arrow);
                }
            }, index * 250); // 250ms stagger
        });

        // Show results
        resultSection.classList.remove('hidden');
    }

    function resetUI() {
        resultSection.classList.add('hidden');
        loadingSection.classList.add('hidden');
        inputSection.classList.remove('hidden');
    }
});
