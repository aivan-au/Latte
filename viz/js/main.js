// Main Application Module for LATTE Viewer

class LatteViewer {
    constructor() {
        this.dataManager = new DataManager();
        this.treeRenderer = new TreeRenderer('#tree-container', this.dataManager);
        this.annotationSystem = new AnnotationSystem(this.treeRenderer, this.dataManager);
        this.sidebar = new SidebarManager(this.dataManager);
        this.sidebarResize = new SidebarResize();
        this.smartTooltips = new SmartTooltips();
        this.controls = new ControlsManager(this.dataManager, this.treeRenderer, this.annotationSystem);
    }
    
    // Initialize the application
    init() {
        // Initialize all components
        this.treeRenderer.init();
        this.annotationSystem.init();
        this.sidebar.init();
        this.sidebarResize.init();
        this.smartTooltips.init();
        this.controls.init();
        
        // Set up node click handling
        this.treeRenderer.onNodeClick = (clusterData) => {
            const searchTerm = this.controls.currentSearchTerm || '';
            this.sidebar.displayCluster(clusterData, searchTerm);
        };
        

    }
}

// Application entry point
window.addEventListener('load', () => {
    // Create global instance for cross-module communication
    window.latteViewer = new LatteViewer();
    window.latteViewer.init();
}); 