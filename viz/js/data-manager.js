// Data Management Module for LATTE Viewer

class DataManager {
    constructor() {
        this.data = null;
        this.currentNodes = null;
    }
    
    // File loading logic
    loadFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    if (this.validateData(data)) {
                        this.data = data;
                        resolve(data);
                    } else {
                        reject(new Error('Invalid data format'));
                    }
                } catch (error) {
                    reject(new Error('Error parsing JSON file: ' + error.message));
                }
            };
            reader.onerror = () => reject(new Error('Error reading file'));
            reader.readAsText(file);
        });
    }
    
    // Embedded data loading
    loadEmbeddedData(embeddedData) {
        if (this.validateData(embeddedData)) {
            this.data = embeddedData;
            return true;
        }
        return false;
    }
    
    // Data validation
    validateData(data) {
        return data && 
               data.metadata && 
               data.clusters && 
               Array.isArray(data.clusters) &&
               data.titles &&
               Array.isArray(data.titles);
    }
    
    // Get current data
    getData() {
        return this.data;
    }
    
    // Get metadata
    getMetadata() {
        return this.data ? this.data.metadata : null;
    }
    
    // Get clusters
    getClusters() {
        return this.data ? this.data.clusters : null;
    }
    
    // Get titles
    getTitles() {
        return this.data ? this.data.titles : null;
    }
    
    // Store current nodes (used by annotation system)
    setCurrentNodes(nodes) {
        this.currentNodes = nodes;
    }
    
    // Get current nodes
    getCurrentNodes() {
        return this.currentNodes;
    }
    
    // Check if data is loaded
    hasData() {
        return this.data !== null;
    }

    // Search for clusters containing documents with matching titles
    searchClusters(searchTerm) {
        if (!this.data || !searchTerm) {
            return { matchingClusters: new Set(), matchingTitleIndices: new Set() };
        }

        const normalizedSearchTerm = searchTerm.toLowerCase();
        const matchingTitleIndices = new Set();
        const matchingClusters = new Set();

        // Find matching title indices
        this.data.titles.forEach((title, index) => {
            if (title.toLowerCase().includes(normalizedSearchTerm)) {
                matchingTitleIndices.add(index);
            }
        });

        // Find clusters that contain any of the matching titles
        this.data.clusters.forEach(cluster => {
            if (cluster.point_indices && cluster.point_indices.some(index => matchingTitleIndices.has(index))) {
                matchingClusters.add(cluster.id);
            }
        });

        return { matchingClusters, matchingTitleIndices };
    }
} 