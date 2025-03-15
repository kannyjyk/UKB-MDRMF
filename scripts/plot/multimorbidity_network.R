# Load necessary packages
library(data.table)
library(dplyr)
library(igraph)
library(ggraph)
library(gridExtra)

selected_methed = "Pred"

# ----------------------------------------------------------------------------
# Read data
if (selected_methed == "Pred") {
  multimor_data <- fread("./results/cache/multimorbidity/multimorbidity_pred.csv")
} else if (selected_methed == "Surv") {
  multimor_data <- fread("./results/cache/multimorbidity/multimorbidity_surv.csv")
}

# Check data dimensions
print(dim(multimor_data))  # For 'Pred', it should be 1560 x 201; for 'Surv', it should be 1560 x 301

# The first column of multimor_data contains Phecode names, the remaining columns are 100-dimensional vectors
phecode_names <- multimor_data$Phecode  # Extract Phecode names
multimor_matrix <- as.matrix(multimor_data[, -1, with = FALSE])  # 100-dimensional vector matrix

#### ------------

# Set correlation threshold
if (selected_methed == "Pred") {
  threshold <- 0.39  # For 'Pred'
} else if (selected_methed == "Surv") {
  threshold <- 0.60  # For 'Surv'
}

# Compute the correlation matrix
cor_matrix <- cor(t(multimor_matrix))

# Apply the threshold
cor_matrix[cor_matrix < threshold] <- 0  # Set values below the threshold to 0

# Build an undirected graph
graph <- graph_from_adjacency_matrix(cor_matrix, mode = "undirected", weighted = TRUE, diag = FALSE)

# Set node names to Phecode names
V(graph)$name <- phecode_names

# Remove isolated nodes (nodes with no connections)
graph <- delete_vertices(graph, which(degree(graph) == 0))

# Perform community detection using the Louvain method
clusters <- cluster_louvain(graph)

# Check community distribution
print(sizes(clusters))

# Extract the top 6 communities
top_communities <- order(sizes(clusters), decreasing = TRUE)[1:6]

#### ------------------------ Add different colors -----------------------------
phecat_data <- read.csv("./results/cache/multimorbidity/phecat_rows_and_categories.csv")

#######
# Create a mapping data frame
phecode_mapping <- data.frame(
  Phecode = phecode_names,
  Row_Number = (1:length(phecode_names)) - 1 # Assuming Row_Number is the integer part of Phecode
)

# Match Phecode to categories
matched_data <- merge(phecode_mapping, phecat_data, by = "Row_Number", all.x = TRUE)
write.csv(matched_data, "matched_data.csv", row.names = FALSE)

###### Plot
# Define category-to-color mapping and remove NA
categories <- unique(na.omit(matched_data$Category))
color_list <- c('#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', 
                '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', 
                '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', 
                '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', 
                '#1b9e77')
category_colors <- setNames(color_list, categories)
category_colors["Other"] <- "#9ACD32"  # Assign a different color for "Other" category
matched_data$Category[is.na(matched_data$Category)] <- "Other"

# Ensure matched_data Phecodes align with node names
matched_data$Phecode <- as.character(matched_data$Phecode)

if (selected_methed == "Pred") {
  # Titles for 'Pred'
  community_titles <- c(
    "Pregnancy and Reproductive System Disorders",
    "Multi-System Interactions",
    "Endocrine Disorders",
    "Genital Disorders",
    "Urinary-Related Disorders",
    "Mental-Related Disorders"
  )
} else if (selected_methed == "Surv") {
  # Titles for 'Surv'
  community_titles <- c(
    "Genital-Related Disorders",
    "Eye Diseases and Vision-Related Disorders",
    "Mental-Related Disorders",
    "Female Reproductive System Disorders",
    "Bone and Soft Tissue Disorders",
    "Ear-Related Disorders"
  )
}

# Create subplots for each community
community_plots <- lapply(seq_along(top_communities), function(i) {
  community <- top_communities[i]
  # Extract subgraph for the current community
  community_nodes <- V(graph)[clusters$membership == community]
  community_subgraph <- induced_subgraph(graph, community_nodes)
  
  # Get Phecodes and categories in the subgraph
  subgraph_phecodes <- V(community_subgraph)$name
  subgraph_categories <- matched_data$Category[match(subgraph_phecodes, matched_data$Phecode)]
  
  # Add category information to subgraph nodes
  V(community_subgraph)$category <- subgraph_categories
  
  # Plot using ggraph
  ggraph(community_subgraph, layout = "fr") +  # Fruchterman-Reingold layout
    geom_edge_link(edge_alpha = 0.3, color = "grey", show.legend = FALSE) +
    geom_node_point(aes(color = category), alpha = 1, show.legend = TRUE) +
    geom_node_text(aes(label = name), repel = TRUE, size = 2.5, color = "black") +  # Reduce text size
    scale_color_manual(values = category_colors) +  # Unify category colors
    labs(
      title = community_titles[i],  # Replace with custom titles
      color = ""
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 10),  # Center title and use smaller font
      legend.box = "horizontal",  # Horizontal legend layout
      legend.spacing.x = unit(2, "mm"),  # Reduce horizontal spacing in legend
      legend.spacing.y = unit(1, "mm"),  # Reduce vertical spacing in legend
      legend.key.size = unit(3, "mm"),   # Reduce legend key size
      legend.title = element_text(size = 10),  # Legend title font size
      legend.text = element_text(size = 8)    # Legend text font size
    )
})

# Combine plots using gridExtra::grid.arrange
combined_plot <- do.call(grid.arrange, c(community_plots, ncol = 2))
threshold_str <- sprintf("%03d", threshold * 100)

# Save the plots
if (selected_methed == "Pred") {
  ## For 'Pred'
  file_path0 <- sprintf("./results/cache/multimorbidity/top6_community_plot_th%s", threshold_str)
  ggsave(paste0(file_path0, ".pdf"),
         plot = combined_plot, width = 9, height = 7)
  ggsave(paste0(file_path0, ".png"),
         plot = combined_plot, width = 9, height = 7)
} else if (selected_methed == "Surv") {
  ## For 'Surv'
  file_path0 <- sprintf("./results/cache/multimorbidity/top6_community_plot_th%s_surv", threshold_str)
  ggsave(paste0(file_path0, ".pdf"),
         plot = combined_plot, width = 9, height = 7)
  ggsave(paste0(file_path0, ".png"),
         plot = combined_plot, width = 9, height = 7)
}
