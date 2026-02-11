// Simple form submission script

async function continueInterview() {
    const statusMsg = document.getElementById('statusMessage');
    const btn = document.querySelector('.primary-btn');
    
    // Animate button to loading state
    btn.classList.add('loading');
    btn.innerHTML = '<span class="btn-spinner"></span> Saving your information...';
    statusMsg.textContent = '';
    
    try {
        // Collect stressors from checkboxes
        const stressors = [];
        document.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
            stressors.push(cb.value);
        });

        // Collect form data using element IDs
        const formData = {
            name: document.getElementById('name')?.value || null,
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
            btn.innerHTML = '<span style="margin-right:6px;">✓</span> Success!';
            statusMsg.textContent = 'Redirecting to interview...';
            statusMsg.style.color = '#10b981';
            
            // Show page transition overlay
            const transition = document.getElementById('pageTransition');
            if (transition) transition.classList.add('active');
            
            // Redirect to interview page with a smooth delay
            setTimeout(() => {
                window.location.href = "/interview";
            }, 1200);
        } else {
            btn.classList.remove('loading');
            btn.innerHTML = 'Continue to Interview  →';
            statusMsg.textContent = 'Error: ' + (result.detail || 'Submission failed');
            statusMsg.style.color = '#ef4444';
            console.error("Form submission failed:", result);
        }
    } catch (error) {
        btn.classList.remove('loading');
        btn.innerHTML = 'Continue to Interview  →';
        statusMsg.textContent = 'Error: ' + error.message;
        statusMsg.style.color = '#ef4444';
        console.error("Error:", error);
    }
}
