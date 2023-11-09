# Load required library
library("TreeDist")
library("ape")

# Set the directory where the njtree files are located
njtree_folder <- "/Users/cassie/Desktop/PlmPlylo/gaptree/"

# Get a list of all files in the directory with extension ".nwk"
njtree_files <- list.files(path = njtree_folder, pattern = "^PF00271.*\\.nwk$", full.names = TRUE)

# Read the true tree
truetree <- "/Users/cassie/Desktop/PlmPlylo/trees/PF00271.tree"
true_tree <- ape::read.tree(truetree)

# Create an empty list to store the scores
scores_list <- list()
similarity_list <- list()

# Loop through each njtree file and calculate the score
for (njtree_file in njtree_files) {
  # Read the njtree
  njtree <- ape::read.tree(njtree_file)
  
  # Calculate the score
  score <- SharedPhylogeneticInfo(true_tree, njtree)
  # Calculate the similarity
  simi <- RobinsonFoulds(true_tree, njtree, similarity = TRUE, normalize = TRUE)
  
  # Store the score in the list with the corresponding file name
  file_name <- basename(njtree_file)
  scores_list[[file_name]] <- score
  similarity_list[[file_name]] <- simi
}

# Print or manipulate the scores as needed
# print(scores_list)
print(similarity_list)

