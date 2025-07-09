// Utility functions for LATTE Viewer

// Color utilities
function rgbaToString(r, g, b, a) {
    return `rgba(${r}, ${g}, ${b}, ${a})`;
}

// Text utilities  
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
}

// Math utilities
function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

// Node color calculation utility
function getNodeColor(cluster, showMask) {
    if (!showMask) {
        return "white"; // Show white when toggle is off
    }
    
    // Check if mask_normalized_proportion exists
    if (cluster.mask_normalized_proportion === undefined) {
        return "white"; // Default to white if no mask data
    }
    
    const normalizedProp = cluster.mask_normalized_proportion;
    
    if (normalizedProp >= 0) {
        // Positive values: use red with alpha proportional to value
        const alpha = normalizedProp; // 0 to 1
        return rgbaToString(239, 68, 68, alpha); // New red color with alpha
    } else {
        // Negative values: use blue with alpha proportional to absolute value
        const alpha = Math.abs(normalizedProp); // 0 to 1 (taking absolute value)
        return rgbaToString(79, 70, 229, alpha); // New blue color with alpha
    }
} 