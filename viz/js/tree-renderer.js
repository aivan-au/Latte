// Tree Renderer Module for LATTE Viewer

class TreeRenderer {
    constructor(containerSelector, dataManager) {
        this.containerSelector = containerSelector;
        this.dataManager = dataManager;
        this.svg = null;
        this.g = null;
        this.width = 0;
        this.height = 0;
        this.selectedNodeId = null;
    }
    
    // Initialize the SVG and tree container
    init() {
        const container = d3.select(this.containerSelector);
        const containerRect = container.node().getBoundingClientRect();
        this.width = containerRect.width;
        this.height = containerRect.height;
        
        this.svg = container.append("svg")
            .attr("width", this.width)
            .attr("height", this.height);
        
        this.g = this.svg.append("g");
        
        // Add zoom functionality
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .filter((event) => {
                // Allow zoom/pan only when not clicking on nodes
                return !event.target.closest('circle');
            })
            .on("zoom", (event) => {
                this.g.attr("transform", event.transform);
            });
        
        this.svg.call(this.zoom);
    }
    
    // Build tree structure from flat cluster list
    buildTree(clusters) {
        const clusterMap = new Map();
        
        // Create a map of all clusters
        clusters.forEach(cluster => {
            clusterMap.set(cluster.id, {
                ...cluster,
                children: []
            });
        });
        
        // Build parent-child relationships
        let root = null;
        clusters.forEach(cluster => {
            if (cluster.parent_id === null) {
                root = clusterMap.get(cluster.id);
            } else {
                const parent = clusterMap.get(cluster.parent_id);
                if (parent) {
                    parent.children.push(clusterMap.get(cluster.id));
                }
            }
        });
        
        return root;
    }
    
    // Main rendering function - render once, update with CSS classes
    render() {
        const data = this.dataManager.getData();
        if (!data) return;
        
        // Only render if not already rendered
        if (this.g.selectAll('.node').size() > 0) {
            console.log('Tree already rendered, skipping re-render');
            return this.dataManager.getCurrentNodes();
        }
        
        // Reset annotation group reference
        this.annotationGroup = null;
        
        // Build tree structure
        const root = this.buildTree(data.clusters);
        if (!root) {
            console.error('No root node found');
            return;
        }
        
        // Create tree layout
        const treeLayout = d3.tree()
            .size([this.width - 100, this.height - 100])
            .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);
        
        // Create hierarchy
        const hierarchy = d3.hierarchy(root);
        const treeData = treeLayout(hierarchy);
        
        // Get nodes and links
        const nodes = treeData.descendants();
        const links = treeData.descendants().slice(1);
        
        // Store current nodes in data manager
        this.dataManager.setCurrentNodes(nodes);
        
        // Calculate bounds to center the tree
        const xExtent = d3.extent(nodes, d => d.x);
        const yExtent = d3.extent(nodes, d => d.y);
        
        const treeWidth = xExtent[1] - xExtent[0];
        
        // Get sidebar width to account for available space
        const sidebarWidth = this.getSidebarWidth();
        const horizontalPadding = 80; // Add padding on both sides for node visibility
        const availableWidth = this.width - sidebarWidth - horizontalPadding;
        
        // Center the tree in the available space (excluding sidebar and padding)
        const translateX = (availableWidth - treeWidth) / 2 - xExtent[0] + (horizontalPadding / 2);
        const translateY = 50;
        
        // Create initial transform and apply it to both the group and the zoom behavior
        const initialTransform = d3.zoomIdentity.translate(translateX, translateY);
        this.g.attr("transform", initialTransform);
        
        // Update the zoom behavior to know about this initial transform
        if (this.svg && this.svg.node().__zoom) {
            this.svg.call(this.zoom.transform, initialTransform);
        }
        
        // Create ALL links (no filtering)
        this.renderAllLinks(links);
        
        // Create ALL nodes (no filtering)
        this.renderAllNodes(nodes);
        
        return nodes;
    }
    
    // Render ALL tree links (no filtering)
    renderAllLinks(links) {
        this.g.selectAll(".link")
            .data(links)
            .enter().append("path")
            .attr("class", d => `link level-${d.data.level}`)
            .attr("data-level", d => d.data.level)
            .attr("data-parent-level", d => d.parent.data.level)
            .attr("d", d => {
                return `M${d.x},${d.y}C${d.x},${(d.y + d.parent.y) / 2} ${d.parent.x},${(d.y + d.parent.y) / 2} ${d.parent.x},${d.parent.y}`;
            });
    }
    
    // Render ALL tree nodes (no filtering)
    renderAllNodes(nodes) {
        // Calculate size scaling based on full dataset
        const clusterSizes = nodes.map(n => n.data.size);
        const minSize = Math.min(...clusterSizes);
        const maxSize = Math.max(...clusterSizes);
        
        // Node radius calculation function
        const getNodeRadius = (size) => {
            if (minSize === maxSize) {
                // All clusters same size, use middle radius
                return 10;
            }
            // Scale between 5px (min) and 15px (max) based on relative position
            const normalizedSize = (size - minSize) / (maxSize - minSize);
            return 5 + (normalizedSize * 10); // 5 + (0 to 1) * 10 = 5 to 15
        };
        
        const node = this.g.selectAll(".node")
            .data(nodes)
            .enter().append("g")
            .attr("class", d => `node level-${d.data.level}`)
            .attr("data-level", d => d.data.level)
            .attr("data-node-id", d => d.data.id)
            .attr("transform", d => `translate(${d.x},${d.y})`);
        
        // Add background circles (solid white)
        node.append("circle")
            .attr("class", "background-circle")
            .attr("data-node-id", d => d.data.id)
            .attr("r", d => getNodeRadius(d.data.size))
            .attr("fill", "white")
            .attr("stroke", "#333")
            .attr("stroke-width", 1.5)
            .style("cursor", "pointer")
            .on("click", (event, d) => {
                // Update selected node
                this.setSelectedNode(d.data.id);
                // Call the click handler if it exists
                if (this.onNodeClick) {
                    this.onNodeClick(d.data);
                }
                // Prevent event bubbling
                event.stopPropagation();
            });
        
        // Add foreground circles (transparent colors)
        node.append("circle")
            .attr("class", "foreground-circle")
            .attr("r", d => getNodeRadius(d.data.size))
            .attr("fill", d => getNodeColor(d.data, true)) // Default to mask view
            .attr("stroke", "none")
            .style("cursor", "pointer")
            .on("mouseover", (event, d) => {
                window.latteViewer.annotationSystem.showTooltip(event, d.data);
            })
            .on("mouseout", () => {
                window.latteViewer.annotationSystem.hideTooltip();
            })
            .on("click", (event, d) => {
                // Update selected node
                this.setSelectedNode(d.data.id);
                // Call the click handler if it exists
                if (this.onNodeClick) {
                    this.onNodeClick(d.data);
                }
                // Prevent event bubbling
                event.stopPropagation();
            });
    }
    
    // Update node colors via CSS class toggle
    updateNodeColors(showMask) {
        if (this.g) {
            this.g.selectAll('.foreground-circle')
                .attr('fill', d => getNodeColor(d.data, showMask));
        }
    }
    
    // Filter nodes by level using CSS classes
    filterByLevel(maxLevel) {
        if (!this.g) return;
        
        // Show/hide nodes based on level
        this.g.selectAll('.node')
            .classed('level-hidden', d => d.data.level < maxLevel);
            
        // Show/hide links based on level (both node and parent must be visible)
        this.g.selectAll('.link')
            .classed('level-hidden', d => d.data.level < maxLevel || d.parent.data.level < maxLevel);
    }
    
    // Show/hide annotations using CSS classes
    toggleAnnotations(show) {
        if (this.annotationGroup) {
            this.annotationGroup.classed('annotations-hidden', !show);
        }
    }

    // Highlight clusters based on search results
    highlightSearch(matchingClusters) {
        if (!this.g) return;

        // Remove all existing search dots
        this.g.selectAll('.search-dot').remove();

        if (matchingClusters.size > 0) {
            // Add search dots to matching nodes
            this.g.selectAll('.node')
                .filter(d => matchingClusters.has(d.data.id))
                .append('circle')
                .attr('class', 'search-dot')
                .attr('r', 3)
                .attr('cx', 0)
                .attr('cy', 0);
        }
    }
    
    // Get SVG group for annotations
    getAnnotationGroup() {
        if (!this.annotationGroup) {
            this.annotationGroup = this.g.append("g").attr("class", "annotations");
        }
        return this.annotationGroup;
    }
    
    // Set selected node and update visuals
    setSelectedNode(nodeId) {
        this.selectedNodeId = nodeId;
        this.updateNodeSelection();
    }
    
    // Clear selected node
    clearSelectedNode() {
        this.selectedNodeId = null;
        this.updateNodeSelection();
    }
    
    // Update node selection visuals
    updateNodeSelection() {
        if (this.g) {
            // Reset all nodes to normal style
            const allCircles = this.g.selectAll('.background-circle');
            allCircles.classed('selected', false);
            
            // Highlight the selected node
            if (this.selectedNodeId !== null) {
                const selectedCircle = this.g.selectAll(`.background-circle[data-node-id="${this.selectedNodeId}"]`);
                selectedCircle.classed('selected', true);
            }
        }
    }
    
    // Get current sidebar width for layout calculations
    getSidebarWidth() {
        try {
            // Try to get width from sidebar resize manager if available
            if (window.latteViewer && window.latteViewer.sidebarResize) {
                return window.latteViewer.sidebarResize.getWidth();
            }
            
            // Fallback to CSS computed width
            const sidebar = document.querySelector('.right-sidebar');
            if (sidebar) {
                const computedStyle = window.getComputedStyle(sidebar);
                return parseInt(computedStyle.width, 10) || 300;
            }
            
            // Final fallback
            return 300;
        } catch (e) {
            console.warn('Could not get sidebar width:', e);
            return 300;
        }
    }
    
    // Reset tree renderer for new data
    reset() {
        if (this.g) {
            // Clear all tree elements including search dots
            this.g.selectAll("*").remove();
        }
        
        // Reset annotation group reference
        this.annotationGroup = null;
        
        // Clear selected node
        this.selectedNodeId = null;
        
        console.log('Tree renderer reset for new data');
    }
} 