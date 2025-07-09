// Annotation System Module for LATTE Viewer

class AnnotationSystem {
    constructor(treeRenderer, dataManager) {
        this.treeRenderer = treeRenderer;
        this.dataManager = dataManager;
        this.tooltip = null;
        this.annotationGroup = null;
        this.isVisible = false;
    }
    
    // Initialize tooltip element
    init() {
        this.tooltip = d3.select("#tooltip");
    }
    
    // Show/hide all annotations
    toggle(show, maxLevel = null) {
        this.isVisible = show;
        if (show) {
            this.showAll(maxLevel);
        } else {
            this.hideAll();
        }
    }
    
    // Create annotation display - render once, filter with CSS
    showAll(maxLevel = null) {
        const nodes = this.dataManager.getCurrentNodes();
        
        if (!nodes) {
            return;
        }
        
        // Get or create annotation group
        this.annotationGroup = this.treeRenderer.getAnnotationGroup();
        
        // Only render if not already rendered
        if (this.annotationGroup.selectAll('.annotation-label').size() > 0) {
            console.log('Annotations already rendered, using CSS filtering');
            this.filterAnnotationsByLevel(maxLevel);
            return;
        }
        
        // IMPORTANT: Calculate positions based on ALL nodes with annotations
        // This ensures consistent positioning regardless of level filtering
        const allNodesWithAnnotations = nodes.filter(d => d.data.annotation);
        
        if (allNodesWithAnnotations.length === 0) {
            return;
        }
        
        // Calculate positioning strategy based on FULL tree size
        const nodeCount = allNodesWithAnnotations.length;
        const yExtent = d3.extent(nodes, d => d.y);
        
        // Use conservative positioning for small trees
        const isSmallTree = nodeCount <= 5;
        const padding = isSmallTree ? 30 : 100;
        const minY = yExtent[0] - padding;
        const maxY = yExtent[1] + padding;
        const availableHeight = maxY - minY;
        
        // Create label data with size-appropriate positioning strategy
        const labelData = [];
        
        allNodesWithAnnotations.forEach((d, nodeIndex) => {
            const text = truncateText(d.data.annotation, 40);
            const width = this.estimateTextWidth(text);
            const height = 20;
            
            let xOffset, yOffset;
            
            // Special handling for root node (highest level, depth 0)
            if (d.depth === 0 || d.data.level === Math.max(...allNodesWithAnnotations.map(n => n.data.level))) {
                // Root annotation: place it close above the root node, always visible
                xOffset = d.x;
                yOffset = d.y - 40; // Fixed 40px above the root node
            } else if (isSmallTree) {
                // For small trees: keep annotations close to nodes
                const sideOffset = (nodeIndex % 2 === 0 ? -1 : 1) * (width / 2 + 20);
                xOffset = d.x + sideOffset;
                yOffset = d.y + ((nodeIndex % 4 < 2) ? -30 : 30); // Alternate above/below
            } else {
                // For larger trees: use more complex positioning
                const nodesByLevel = d3.group(allNodesWithAnnotations, d => d.depth);
                const levels = Array.from(nodesByLevel.keys()).sort((a, b) => a - b);
                const levelIndex = levels.indexOf(d.depth);
                const levelNodes = nodesByLevel.get(d.depth);
                const nodeIndexInLevel = levelNodes.indexOf(d);
                
                const baseY = minY + (levelIndex / (levels.length - 1 || 1)) * availableHeight;
                const ySpread = availableHeight / (levels.length * 2);
                const yOffsetInLevel = (nodeIndexInLevel - levelNodes.length / 2) * (ySpread / levelNodes.length);
                const randomOffset = (Math.random() - 0.5) * 40;
                
                const sideOffset = (nodeIndex % 2 === 0 ? -1 : 1) * (80 + Math.random() * 40);
                xOffset = d.x + sideOffset;
                yOffset = baseY + yOffsetInLevel + randomOffset;
            }
            
            labelData.push({
                id: d.data.id,
                text: text,
                nodeX: d.x,
                nodeY: d.y,
                x: xOffset,
                y: yOffset,
                width: width,
                height: height,
                node: d,
                level: d.depth
            });
        });
        
        // Apply enhanced collision resolution (excluding root node)
        this.resolveCollisions(labelData, minY, maxY);
        
        // Create ALL connector lines (no filtering)
        this.annotationGroup.selectAll(".annotation-connector")
            .data(labelData)
            .enter().append("line")
            .attr("class", d => `annotation-connector level-${d.node.data.level}`)
            .attr("data-level", d => d.node.data.level)
            .attr("x1", d => d.nodeX)
            .attr("y1", d => d.nodeY)
            .attr("x2", d => d.x)
            .attr("y2", d => d.y);
        
        // Create ALL label backgrounds (rectangles)
        this.annotationGroup.selectAll(".annotation-background")
            .data(labelData)
            .enter().append("rect")
            .attr("class", d => `annotation-background level-${d.node.data.level}`)
            .attr("data-level", d => d.node.data.level)
            .attr("x", d => d.x - (d.width / 2))
            .attr("y", d => d.y - 10)
            .attr("width", d => d.width)
            .attr("height", d => d.height)
            .attr("rx", 4)
            .style("cursor", "pointer")
            .on("mouseover", (event, d) => {
                // Show full annotation text on hover
                this.showTooltip(event, { annotation: d.node.data.annotation });
            })
            .on("mouseout", () => {
                this.hideTooltip();
            });
        
        // Create ALL labels
        this.annotationGroup.selectAll(".annotation-label")
            .data(labelData)
            .enter().append("text")
            .attr("class", d => `annotation-label level-${d.node.data.level}`)
            .attr("data-level", d => d.node.data.level)
            .attr("x", d => d.x)
            .attr("y", d => d.y + 4)
            .text(d => d.text)
            .style("cursor", "pointer")
            .on("mouseover", (event, d) => {
                // Show full annotation text on hover
                this.showTooltip(event, { annotation: d.node.data.annotation });
            })
            .on("mouseout", () => {
                this.hideTooltip();
            });
            
        // Apply initial level filtering if specified
        if (maxLevel !== null) {
            this.filterAnnotationsByLevel(maxLevel);
        }
    }
    
    // Enhanced collision resolution using iterative refinement
    resolveCollisions(labelData, minY, maxY) {
        const maxIterations = 100; // Increased iterations
        const margin = 8; // Increased margin for better spacing
        
        // Identify root node (don't move it during collision resolution)
        const rootLabel = labelData.find(label => 
            label.node.depth === 0 || 
            label.node.data.level === Math.max(...labelData.map(l => l.node.data.level))
        );
        
        for (let iteration = 0; iteration < maxIterations; iteration++) {
            let hasCollisions = false;
            let totalMoves = 0;
            
            // Sort labels by priority (smaller labels get moved first, root stays put)
            labelData.sort((a, b) => {
                if (a === rootLabel) return -1; // Root has highest priority (don't move)
                if (b === rootLabel) return 1;
                return a.width - b.width;
            });
            
            // Check all pairs for collisions
            for (let i = 0; i < labelData.length; i++) {
                for (let j = i + 1; j < labelData.length; j++) {
                    const labelA = labelData[i];
                    const labelB = labelData[j];
                    
                    if (this.rectanglesOverlap(labelA, labelB, margin)) {
                        hasCollisions = true;
                        totalMoves += this.separateLabels(labelA, labelB, margin, minY, maxY, rootLabel);
                        
                        // If we made significant moves, check this pair again
                        if (totalMoves > 2) {
                            j = i; // Restart inner loop to recheck this label against all others
                            totalMoves = 0;
                        }
                    }
                }
            }
            
            // If no collisions found, we're done
            if (!hasCollisions) {
                break;
            }
            
            // Add some randomization to avoid getting stuck in local minima
            if (iteration > 50 && hasCollisions) {
                this.addRandomJitter(labelData, 2, minY, maxY, rootLabel);
            }
        }
    }
    
    // Add small random movements to break out of local minima
    addRandomJitter(labelData, amount, minY, maxY, rootLabel = null) {
        labelData.forEach(label => {
            // Don't jitter the root label
            if (label === rootLabel) return;
            
            label.x += (Math.random() - 0.5) * amount;
            label.y += (Math.random() - 0.5) * amount;
            
            // Keep within bounds
            const maxWidth = this.treeRenderer.width;
            label.x = Math.max(label.width / 2 + 10, Math.min(maxWidth - label.width / 2 - 10, label.x));
            label.y = Math.max(minY + 20, Math.min(maxY - 20, label.y));
        });
    }
    
    // Check if two rectangular labels overlap (improved)
    rectanglesOverlap(labelA, labelB, margin = 0) {
        const aLeft = labelA.x - labelA.width / 2 - margin;
        const aRight = labelA.x + labelA.width / 2 + margin;
        const aTop = labelA.y - labelA.height / 2 - margin;
        const aBottom = labelA.y + labelA.height / 2 + margin;
        
        const bLeft = labelB.x - labelB.width / 2 - margin;
        const bRight = labelB.x + labelB.width / 2 + margin;
        const bTop = labelB.y - labelB.height / 2 - margin;
        const bBottom = labelB.y + labelB.height / 2 + margin;
        
        return !(aRight <= bLeft || aLeft >= bRight || aBottom <= bTop || aTop >= bBottom);
    }
    
    // Separate two overlapping labels (improved)
    separateLabels(labelA, labelB, margin, minY, maxY, rootLabel = null) {
        const dx = labelB.x - labelA.x;
        const dy = labelB.y - labelA.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 0.1) {
            // If labels are at exactly the same position, separate them arbitrarily
            const angle = Math.random() * 2 * Math.PI;
            labelB.x += Math.cos(angle) * 60;
            labelB.y += Math.sin(angle) * 30;
            this.keepInBounds(labelB, minY, maxY);
            return 2; // Significant move
        }
        
        // Calculate required separation distance
        const requiredSeparationX = (labelA.width + labelB.width) / 2 + margin;
        const requiredSeparationY = (labelA.height + labelB.height) / 2 + margin;
        
        // Calculate current separation
        const currentSeparationX = Math.abs(dx);
        const currentSeparationY = Math.abs(dy);
        
        // Calculate how much we need to move in each direction
        const overlapX = Math.max(0, requiredSeparationX - currentSeparationX);
        const overlapY = Math.max(0, requiredSeparationY - currentSeparationY);
        
        let moveCount = 0;
        
        // Move labels apart based on the overlap amounts
        if (overlapX > 0 || overlapY > 0) {
            // Prefer moving in the direction with less overlap (easier to resolve)
            if (overlapX <= overlapY || overlapY === 0) {
                // Separate horizontally
                const moveDistance = (overlapX + 2) / 2; // Add extra spacing
                if (dx >= 0) {
                    if (labelA !== rootLabel) labelA.x -= moveDistance;
                    if (labelB !== rootLabel) labelB.x += moveDistance;
                    // If one is root, move the other twice as far
                    if (labelA === rootLabel && labelB !== rootLabel) labelB.x += moveDistance;
                    if (labelB === rootLabel && labelA !== rootLabel) labelA.x -= moveDistance;
                } else {
                    if (labelA !== rootLabel) labelA.x += moveDistance;
                    if (labelB !== rootLabel) labelB.x -= moveDistance;
                    // If one is root, move the other twice as far
                    if (labelA === rootLabel && labelB !== rootLabel) labelB.x -= moveDistance;
                    if (labelB === rootLabel && labelA !== rootLabel) labelA.x += moveDistance;
                }
                moveCount = 1;
            }
            
            if (overlapY > 0 && (overlapY < overlapX || overlapX === 0)) {
                // Separate vertically
                const moveDistance = (overlapY + 2) / 2; // Add extra spacing
                if (dy >= 0) {
                    if (labelA !== rootLabel) labelA.y -= moveDistance;
                    if (labelB !== rootLabel) labelB.y += moveDistance;
                    // If one is root, move the other twice as far
                    if (labelA === rootLabel && labelB !== rootLabel) labelB.y += moveDistance;
                    if (labelB === rootLabel && labelA !== rootLabel) labelA.y -= moveDistance;
                } else {
                    if (labelA !== rootLabel) labelA.y += moveDistance;
                    if (labelB !== rootLabel) labelB.y -= moveDistance;
                    // If one is root, move the other twice as far
                    if (labelA === rootLabel && labelB !== rootLabel) labelB.y -= moveDistance;
                    if (labelB === rootLabel && labelA !== rootLabel) labelA.y += moveDistance;
                }
                moveCount = Math.max(moveCount, 1);
            }
        }
        
        // Keep labels within bounds
        this.keepInBounds(labelA, minY, maxY);
        this.keepInBounds(labelB, minY, maxY);
        
        return moveCount;
    }
    
    // Helper to keep labels within bounds
    keepInBounds(label, minY, maxY) {
        const maxWidth = this.treeRenderer.width;
        label.x = Math.max(label.width / 2 + 10, Math.min(maxWidth - label.width / 2 - 10, label.x));
        label.y = Math.max(minY + 20, Math.min(maxY - 20, label.y));
    }
    
    // Helper method to estimate text width for Roboto Condensed
    estimateTextWidth(text) {
        // Roboto Condensed is narrower, so we can use a smaller multiplier
        return Math.max(60, text.length * 4.5); // Reduced from 6 to 4.5 for condensed font
    }
    
    // Clear annotations
    hideAll() {
        if (this.annotationGroup) {
            this.annotationGroup.selectAll("*").remove();
        }
    }
    
    // Filter annotations by level using CSS classes
    filterAnnotationsByLevel(maxLevel) {
        if (!this.annotationGroup) return;
        
        if (maxLevel === null) {
            // Show all annotations
            this.annotationGroup.selectAll('.annotation-label, .annotation-background, .annotation-connector')
                .classed('level-hidden', false);
        } else {
            // Hide annotations for nodes below the level threshold
            this.annotationGroup.selectAll('.annotation-label, .annotation-background, .annotation-connector')
                .classed('level-hidden', function() {
                    const level = parseInt(d3.select(this).attr('data-level'), 10);
                    return level < maxLevel;
                });
        }
    }
    
    // Tooltip management
    showTooltip(event, cluster) {
        const annotation = cluster.annotation || "No annotation available";
        
        this.tooltip
            .html(annotation)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px")
            .classed("show", true);
    }
    
    hideTooltip() {
        this.tooltip.classed("show", false);
    }
    
    // Get current visibility state
    isAnnotationsVisible() {
        return this.isVisible;
    }
    
    // Reset annotation system for new data
    reset() {
        // Clear all annotations
        if (this.annotationGroup) {
            this.annotationGroup.selectAll("*").remove();
        }
        
        // Reset state
        this.isVisible = false;
        
        console.log('Annotation system reset for new data');
    }
} 