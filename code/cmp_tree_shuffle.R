library(TreeDist)
library(ape)


paths <- list(
  nj_tree_folder = "MsaPhylo/Trees/Pfam/NJ/",
  ml_tree_folder = "MsaPhylo/Trees/Pfam/ML/"
)

prot_domains <- c(
  'PF00066', 'PF00168', 'PF00484', 'PF00672',
  'PF00699', 'PF01951', 'PF03147', 'PF03463',
  'PF04972', 'PF06427', 'PF07498', 'PF07677',
  'PF10114', 'PF10576', 'PF12172', 'PF12872',
  'PF13377', 'PF13466', 'PF14317', 'PF20171'
)


# Function to write scores to files
write_scores <- function(data, file_path) {
  write.table(data, file = file_path, append = TRUE, sep = ",",
              col.names = !file.exists(file_path), row.names = FALSE)
}

# Helper function to process files and calculate scores
process_file <- function(tree_file, nj_tree, ml_tree, result_file) {
  tree <- ape::read.tree(tree_file)
  
  # Calculate scores
  nj_rf_score <- RobinsonFoulds(tree, nj_tree, similarity = TRUE, normalize = TRUE)
  ml_rf_score <- RobinsonFoulds(tree, ml_tree, similarity = TRUE, normalize = TRUE)
  nj_ci_score <- MutualClusteringInfo(tree, nj_tree, normalize = TRUE)
  ml_ci_score <- MutualClusteringInfo(tree, ml_tree, normalize = TRUE)
  
  file_name <- basename(sub("\\.nwk$", "", tree_file))
  data <- data.frame(FileName = file_name,
                     NJRFScore = nj_rf_score,
                     MLRFScore = ml_rf_score,
                     NJCI = nj_ci_score,
                     MLCI = ml_ci_score)
  # Write to the correct result file
  write_scores(data, result_file)
}

calculate_protein_family_scores <- function(folder_name) {
  # Directory paths for embeddings, attentions, and results
  emb_folder <- file.path("/Users/cassie/Desktop/new/embeddings", folder_name)
  attn_folder <- file.path("/Users/cassie/Desktop/new/attentions", folder_name)
  results_folder <- file.path("/Users/cassie/Desktop/new/results", folder_name)
  
  # Corrected file paths for results
  emb_path <- file.path(results_folder, "emb_score.csv")
  attn_path <- file.path(results_folder, "attn_score.csv")
  
  for (prot_domain in prot_domains) {
    pattern_string <- paste0("^", prot_domain, ".*\\.nwk$")
    emb_files <- list.files(path = emb_folder, pattern = pattern_string, full.names = TRUE)
    attn_files <- list.files(path = attn_folder, pattern = pattern_string, full.names = TRUE)
    
    njtree <- file.path(paths$nj_tree_folder, paste0(prot_domain, ".tree"))
    mltree <- file.path(paths$ml_tree_folder, paste0(prot_domain, ".tree"))
    
    nj_tree <- ape::read.tree(njtree)
    ml_tree <- ape::read.tree(mltree)
    
    for (emb_file in emb_files) {
      process_file(emb_file, nj_tree, ml_tree, emb_path)
    }
    
    for (attn_file in attn_files) {
      process_file(attn_file, nj_tree, ml_tree, attn_path)
    }
  }
}

# Run the function for a specific folder
calculate_protein_family_scores("rep5")
