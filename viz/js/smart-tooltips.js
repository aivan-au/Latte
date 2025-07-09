// Smart Tooltips Module for LATTE Viewer

class SmartTooltips {
    constructor() {
        this.tooltip = null;
        this.hideTimeout = null;
    }
    
    // Initialize the tooltip system
    init() {
        this.createTooltipElement();
        this.attachEventListeners();
    }
    
    // Create the tooltip DOM element
    createTooltipElement() {
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'smart-tooltip';
        document.body.appendChild(this.tooltip);
    }
    
    // Attach event listeners to elements with title attributes
    attachEventListeners() {
        // Target elements that should show tooltips
        const tooltipElements = document.querySelectorAll('.icon-toggle, .file-input-label, .level-slider');
        
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', (e) => this.showTooltip(e));
            element.addEventListener('mouseleave', () => this.hideTooltip());
            element.addEventListener('mousemove', (e) => this.updatePosition(e));
        });
    }
    
    // Show tooltip with smart positioning
    showTooltip(event) {
        const element = event.currentTarget;
        const text = element.getAttribute('title') || element.getAttribute('data-tooltip');
        
        if (!text) return;
        
        // Clear any existing hide timeout
        if (this.hideTimeout) {
            clearTimeout(this.hideTimeout);
            this.hideTimeout = null;
        }
        
        this.tooltip.textContent = text;
        this.tooltip.classList.add('show');
        
        this.updatePosition(event);
    }
    
    // Update tooltip position with smart viewport detection
    updatePosition(event) {
        if (!this.tooltip.classList.contains('show')) return;
        
        const element = event.currentTarget;
        const rect = element.getBoundingClientRect();
        const tooltipRect = this.tooltip.getBoundingClientRect();
        const viewport = {
            width: window.innerWidth,
            height: window.innerHeight
        };
        
        let left, top;
        const offset = 8; // Distance from element
        
        // Default position: below and centered
        left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
        top = rect.bottom + offset;
        
        // Check if tooltip would go outside viewport horizontally
        if (left < 0) {
            // Too far left, align to left edge
            left = offset;
        } else if (left + tooltipRect.width > viewport.width) {
            // Too far right, align to right edge
            left = viewport.width - tooltipRect.width - offset;
        }
        
        // Check if tooltip would go outside viewport vertically
        if (top + tooltipRect.height > viewport.height) {
            // Show above element instead
            top = rect.top - tooltipRect.height - offset;
            
            // If still outside viewport, show at top
            if (top < 0) {
                top = offset;
            }
        }
        
        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
    }
    
    // Hide tooltip with delay
    hideTooltip() {
        this.hideTimeout = setTimeout(() => {
            this.tooltip.classList.remove('show');
        }, 100); // Small delay to prevent flickering
    }
    
    // Remove tooltip element
    destroy() {
        if (this.tooltip) {
            this.tooltip.remove();
            this.tooltip = null;
        }
        if (this.hideTimeout) {
            clearTimeout(this.hideTimeout);
            this.hideTimeout = null;
        }
    }
} 