document.addEventListener('DOMContentLoaded', () => {
    const fab = document.getElementById('complaintFab');
    const modal = document.getElementById('complaintModal');
    const closeBtn = document.getElementById('complaintClose');
    const form = document.getElementById('complaintForm');
    const successMsg = document.getElementById('complaintSuccess');

    if (fab && modal && closeBtn) {
        fab.addEventListener('click', () => {
            modal.classList.add('active');
            successMsg.style.display = 'none';
            form.style.display = 'block';
            form.reset();
        });

        closeBtn.addEventListener('click', () => {
            modal.classList.remove('active');
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    }

    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            // Simulate form submission
            form.style.display = 'none';
            successMsg.style.display = 'block';

            // Auto close after 3 seconds
            setTimeout(() => {
                if (modal.classList.contains('active')) {
                    modal.classList.remove('active');
                }
            }, 3000);
        });
    }
});
