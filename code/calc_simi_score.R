# Load required library
library("TreeDist")
library("ape")

# Set the directory where the NJtree files are located
emb_folder <- "/Users/cassie/Desktop/PlmPlylo/embeddings"
prot_domains <- c('PF00066','PF00168','PF00484','PF00672',
                 'PF00699','PF01951','PF03147','PF03463',
                 'PF04972','PF06427','PF07498','PF07677',
                 'PF10114','PF10576','PF12172','PF12872',
                 'PF13377','PF13466','PF14317','PF20171')


# Get a list of all files in the directory with extension ".nwk"
for (prot_domain in prot_domains) {
  pattern_string <- paste0("^", prot_domain, ".*\\.nwk$")
  emb_files <- list.files(path = emb_folder, pattern = pattern_string, full.names = TRUE)
  # Load the NJ tree for this protein domain
  njtree <- paste0("/Users/cassie/Desktop/MsaPhylo/Trees/Pfam/NJ/", prot_domain, ".tree")
  nj_tree <- ape::read.tree(njtree)
  # Load the ML tree for this protein domain
  mltree <- paste0("/Users/cassie/Desktop/MsaPhylo/Trees/Pfam/ML/", prot_domain, ".tree")
  ml_tree <- ape::read.tree(mltree)

  # Create an empty list to store the scores
  rf_score_nj_list <- list()
  rf_score_ml_list <- list()

  # Loop through each njtree file and calculate the score
  for (emb_file in emb_files) {

    # Read the embedding trees
    emb_tree <- ape::read.tree(emb_file)

    # Calculate the Robinson Foulds score
    rf_nj_score <- RobinsonFoulds(emb_tree, nj_tree, similarity = TRUE, normalize = TRUE)
    rf_ml_score <- RobinsonFoulds(emb_tree, ml_tree, similarity = TRUE, normalize = TRUE)

    # Store the score in the list with the corresponding file name
    file_name <- basename(sub("\\.nwk$", "", emb_file))
    nj_data <- data.frame(Column1 = file_name, Column2 = rf_nj_score)
    ml_data <- data.frame(Column1 = file_name, Column2 = rf_ml_score)

    nj_file_path <- "/Users/cassie/Desktop/PlmPlylo/embeddings/nj_rf_score.csv"
    ml_file_path <- "/Users/cassie/Desktop/PlmPlylo/embeddings/ml_rf_score.csv"
    write.table(nj_data, file = nj_file_path, append = TRUE, sep = ",", col.names = !file.exists(file_path), row.names = FALSE)
    write.table(ml_data, file = ml_file_path, append = TRUE, sep = ",", col.names = !file.exists(file_path), row.names = FALSE)

  }
}

