// ============================================================================
// script.js - Global JavaScript for Depression Detection EEG Website
// ============================================================================
// Note: Page-specific logic (upload handling, result rendering) lives directly
// inside predict.html and result.html <script> blocks. This file is for
// site-wide behavior shared across every page.
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    console.log('Depression Detection EEG - site scripts loaded');

    // Highlight the active nav link based on current URL
    const currentPath = window.location.pathname;
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active', 'fw-bold');
        }
    });

    // Auto-dismiss any Bootstrap alerts after 6 seconds (optional UX touch)
    document.querySelectorAll('.alert:not(#errorAlert):not(.alert-warning):not(.alert-info)').forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s ease';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 6000);
    });
});