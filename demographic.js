// Simple form submission script

async function continueInterview() {
    const statusMsg = document.getElementById('statusMessage');
    statusMsg.textContent = 'Submitting...';
    statusMsg.style.color = '#94a3b8';
    
    try {
        // Collect stressors from checkboxes
        const stressors = [];
        document.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
            stressors.push(cb.value);
        });

        // Collect form data using element IDs
        const formData = {
            age: document.getElementById('age')?.value || null,
            gender: document.getElementById('gender')?.value || null,
            country: document.getElementById('country')?.value || null,
            role: document.getElementById('role')?.value || null,
            stage: document.getElementById('stage')?.value || null,
            focus: document.getElementById('focus')?.value || null,
            sleep_duration: document.getElementById('sleep_duration')?.value || null,
            workload: document.getElementById('workload')?.value || null,
            screen_time: document.getElementById('screen_time')?.value || null,
            living_situation: document.getElementById('living_situation')?.value || null,
            support_system: document.getElementById('support_system')?.value || null,
            stressors: stressors
        };

        console.log("Submitting form data:", formData);

        // Submit demographic data to backend
        const response = await fetch('/submit_demographics', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();
        console.log("Response:", response.status, result);

        if (response.ok) {
            statusMsg.textContent = 'Success! Redirecting to interview...';
            statusMsg.style.color = '#10b981';
            
            // Redirect to interview page
            setTimeout(() => {
                window.location.href = "/interview";
            }, 500);
        } else {
            statusMsg.textContent = 'Error: ' + (result.detail || 'Submission failed');
            statusMsg.style.color = '#ef4444';
            console.error("Form submission failed:", result);
        }
    } catch (error) {
        statusMsg.textContent = 'Error: ' + error.message;
        statusMsg.style.color = '#ef4444';
        console.error("Error:", error);
    }
}
