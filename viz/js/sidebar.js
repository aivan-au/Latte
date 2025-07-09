// Sidebar Module for LATTE Viewer

class SidebarManager {
    constructor(dataManager) {
        this.dataManager = dataManager;
        this.sidebarElement = document.querySelector('.right-sidebar');
        this.selectedCluster = null;
    }
    
    // Initialize sidebar
    init() {
        this.renderEmptyState();
    }
    
    // Show empty state when no cluster is selected
    renderEmptyState() {
        if (!this.sidebarElement) return;
        
        this.selectedCluster = null;
        
        // Clear node selection in tree renderer
        if (window.latteViewer && window.latteViewer.treeRenderer) {
            window.latteViewer.treeRenderer.clearSelectedNode();
        }
        
        // Preserve the resize handle if it exists
        const resizeHandle = this.sidebarElement.querySelector('.resize-handle');
        
        this.sidebarElement.innerHTML = `
            <div class="sidebar-empty-state">
                <div class="empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#ccc" stroke-width="1">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                </div>
                <p class="empty-text">Click on a node to view cluster contents</p>
            </div>
        `;
        
        // Re-add the resize handle if it existed
        if (resizeHandle) {
            this.sidebarElement.insertBefore(resizeHandle, this.sidebarElement.firstChild);
        }
    }
    
    // Display cluster details and titles
    displayCluster(clusterData, searchTerm = '') {
        if (!this.sidebarElement || !clusterData) return;
        
        this.selectedCluster = clusterData;
        const titles = this.dataManager.getTitles();
        const clusterTitles = this.getClusterTitles(clusterData, titles);
        
        // Limit to 50 random titles if more than 50
        const displayTitles = clusterTitles.length > 50 
            ? this.getRandomSample(clusterTitles, 50)
            : clusterTitles;
        
        const annotation = clusterData.annotation || 'No annotation available';
        
        // Preserve the resize handle if it exists
        const resizeHandle = this.sidebarElement.querySelector('.resize-handle');
        
        this.sidebarElement.innerHTML = `
            <div class="cluster-details">
                <div class="cluster-summary">
                    <p>${this.decodeHtml(annotation)}</p>
                    ${clusterData.mask_proportion !== undefined ? `
                    <div class="mask-proportion">${(clusterData.mask_proportion * 100).toFixed(1)}% mask proportion</div>
                    ` : ''}
                </div>
                
                <div class="cluster-titles">
                    <h4>
                        Documents 
                        ${clusterTitles.length > 50 ? `
                        <span class="sample-note">(showing ${displayTitles.length} of ${clusterTitles.length})</span>
                        ` : `
                        <span class="total-count">(${displayTitles.length})</span>
                        `}
                    </h4>
                    <div class="titles-list">
                        ${displayTitles.map((title, index) => `
                            <div class="title-item">
                                <span class="title-number">${index + 1}.</span>
                                <span class="title-text ${this.isMatchingTitle(title, searchTerm) ? 'search-highlight' : ''}">${this.decodeHtml(title)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        
        // Re-add the resize handle if it existed
        if (resizeHandle) {
            this.sidebarElement.insertBefore(resizeHandle, this.sidebarElement.firstChild);
        }
    }
    
    // Get titles for a specific cluster
    getClusterTitles(clusterData, allTitles) {
        if (!clusterData.point_indices || !allTitles) return [];
        
        return clusterData.point_indices.map(index => allTitles[index]).filter(title => title);
    }
    
    // Get random sample of titles
    getRandomSample(array, sampleSize) {
        const shuffled = [...array].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, sampleSize);
    }

    // Check if title matches search term (case-insensitive)
    isMatchingTitle(title, searchTerm) {
        return searchTerm && title.toLowerCase().includes(searchTerm.toLowerCase());
    }
    
    // Decode HTML entities properly
    decodeHtml(text) {
        const textarea = document.createElement('textarea');
        textarea.innerHTML = text;
        const decoded = textarea.value;
        
        // Now escape for XSS protection
        const div = document.createElement('div');
        div.textContent = decoded;
        return div.innerHTML;
    }
    
    // Get currently selected cluster
    getSelectedCluster() {
        return this.selectedCluster;
    }
    
    // Check if sidebar is showing cluster details
    isShowingCluster() {
        return this.selectedCluster !== null;
    }

    // Refresh current cluster display with new search term
    refreshWithSearch(searchTerm) {
        if (this.selectedCluster) {
            this.displayCluster(this.selectedCluster, searchTerm);
        }
    }
} 