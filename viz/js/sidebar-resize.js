// Sidebar Resize Module for LATTE Viewer

class SidebarResize {
    constructor() {
        this.sidebar = null;
        this.resizeHandle = null;
        this.isResizing = false;
        this.minWidth = 200;
        this.maxWidth = 1200;
        this.currentWidth = 300;
    }
    
    // Initialize the resize functionality
    init() {
        this.sidebar = document.querySelector('.right-sidebar');
        this.resizeHandle = document.querySelector('.resize-handle');
        
        if (!this.sidebar || !this.resizeHandle) {
            console.warn('Sidebar or resize handle not found');
            return;
        }
        
        this.attachEventListeners();
    }
    
    // Attach event listeners for resize functionality
    attachEventListeners() {
        this.resizeHandle.addEventListener('mousedown', this.handleMouseDown.bind(this));
        document.addEventListener('mousemove', this.handleMouseMove.bind(this));
        document.addEventListener('mouseup', this.handleMouseUp.bind(this));
        
        // Prevent text selection during resize
        this.resizeHandle.addEventListener('selectstart', (e) => e.preventDefault());
    }
    
    // Handle mouse down on resize handle
    handleMouseDown(event) {
        this.isResizing = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        
        // Disable CSS transition during resize for smooth dragging
        this.sidebar.style.transition = 'none';
        
        event.preventDefault();
    }
    
    // Handle mouse move during resize
    handleMouseMove(event) {
        if (!this.isResizing) return;
        
        const containerRect = document.querySelector('.main-content').getBoundingClientRect();
        const newWidth = containerRect.right - event.clientX;
        
        // Constrain width within min/max bounds
        const constrainedWidth = Math.max(this.minWidth, Math.min(this.maxWidth, newWidth));
        
        this.setSidebarWidth(constrainedWidth);
        
        event.preventDefault();
    }
    
    // Handle mouse up to stop resizing
    handleMouseUp(event) {
        if (!this.isResizing) return;
        
        this.isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        
        // Re-enable CSS transition
        this.sidebar.style.transition = 'width 0.2s ease';
        
        event.preventDefault();
    }
    
    // Set the sidebar width
    setSidebarWidth(width) {
        this.currentWidth = width;
        this.sidebar.style.width = `${width}px`;
    }
    

    
    // Get current sidebar width
    getWidth() {
        return this.currentWidth;
    }
    
    // Set width programmatically
    setWidth(width) {
        const constrainedWidth = Math.max(this.minWidth, Math.min(this.maxWidth, width));
        this.setSidebarWidth(constrainedWidth);
    }
} 