# Load required libraries
library("TreeDist")
library("ape")

# Set the directory paths
emb_folder <- "/Users/cassie/Desktop/PlmPlylo/embeddings"
nj_tree_folder <- "/Users/cassie/Desktop/MsaPhylo/Trees/Pfam/NJ/"
ml_tree_folder <- "/Users/cassie/Desktop/MsaPhylo/Trees/Pfam/ML/"
prot_domains <- c('PF00066','PF00168','PF00484','PF00672',
                  'PF00699','PF01951','PF03147','PF03463',
                  'PF04972','PF06427','PF07498','PF07677',
                  'PF10114','PF10576','PF12172','PF12872',
                  'PF13377','PF13466','PF14317','PF20171')

# File paths for output
nj_file_path <- "/Users/cassie/Desktop/PlmPlylo/embeddings/nj_rf_score.csv"
ml_file_path <- "/Users/cassie/Desktop/PlmPlylo/embeddings/ml_rf_score.csv"
nj_ml_file_path <- "/Users/cassie/Desktop/PlmPlylo/embeddings/nj_ml_rf_score.csv"

# Check if files exist to determine if column names should be written
nj_file_exists <- file.exists(nj_file_path)
ml_file_exists <- file.exists(ml_file_path)
nj_ml_file_exists <- file.exists(nj_ml_file_path)

# Function to write scores to files
write_scores <- function(data, file_path, file_exists) {
  write.table(data, file = file_path, append = TRUE, sep = ",", col.names = !file_exists, row.names = FALSE)
}

# Loop through each protein domain
for (prot_domain in prot_domains) {
  pattern_string <- paste0("^", prot_domain, ".*\\.nwk$")
  emb_files <- list.files(path = emb_folder, pattern = pattern_string, full.names = TRUE)

  # Construct paths for NJ and ML trees
  njtree <- file.path(nj_tree_folder, paste0(prot_domain, ".tree"))
  mltree <- file.path(ml_tree_folder, paste0(prot_domain, ".tree"))

  # Read the NJ and ML trees
  nj_tree <- ape::read.tree(njtree)
  ml_tree <- ape::read.tree(mltree)

  # Loop through each embedding file
  for (emb_file in emb_files) {
    emb_tree <- ape::read.tree(emb_file)

    # Calculate Robinson Foulds scores
    rf_nj_score <- RobinsonFoulds(emb_tree, nj_tree, similarity = TRUE, normalize = TRUE)
    rf_ml_score <- RobinsonFoulds(emb_tree, ml_tree, similarity = TRUE, normalize = TRUE)

    file_name <- basename(sub("\\.nwk$", "", emb_file))
    nj_data <- data.frame(FileName = file_name, RFScore = rf_nj_score)
    ml_data <- data.frame(FileName = file_name, RFScore = rf_ml_score)

    # Write NJ and ML scores to respective files
    write_scores(nj_data, nj_file_path, nj_file_exists)
    write_scores(ml_data, ml_file_path, ml_file_exists)

    # Update file existence status
    nj_file_exists <- TRUE
    ml_file_exists <- TRUE
  }

  # Calculate and record the Robinson Foulds score between NJ and ML trees
  domain_name <- basename(prot_domain)
  rf_nj_ml_score <- RobinsonFoulds(nj_tree, ml_tree, similarity = TRUE, normalize = TRUE)
  nj_ml_data <- data.frame(ProteinDomain = domain_name, RFScore = rf_nj_ml_score)

  write_scores(nj_ml_data, nj_ml_file_path, nj_ml_file_exists)
  nj_ml_file_exists <- TRUE  # Update file existence status
}


