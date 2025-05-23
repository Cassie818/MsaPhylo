library(TreeDist)
library(ape)

# setting the working directory
paths <- list(
  emb_folder = "MsaPhylo/embeddings",
  attn_folder = "MsaPhylo/attentions",
  nj_tree_folder = "MsaPhylo/Trees/Pfam/NJ/",
  ml_tree_folder = "MsaPhylo/Trees/Pfam/ML/",
  emb_file = "MsaPhylo/results/emb_score.csv",
  attn_file = "MsaPhylo/results/attn_score.csv",
  nj_ml_file = "MsaPhylo/results/nj_ml_score.csv"
)

prot_domains <- c(
  'PF00066', 'PF00168', 'PF00484', 'PF00672',
  'PF00699', 'PF01951', 'PF03147', 'PF03463',
  'PF04972', 'PF06427', 'PF07498', 'PF07677',
  'PF10114', 'PF10576', 'PF12172', 'PF12872',
  'PF13377', 'PF13466', 'PF14317', 'PF20171'
)

write_scores <- function(data, file_path) {
  write.table(data, file = file_path, append = TRUE, sep = ",",
              col.names = !file.exists(file_path), row.names = FALSE)
}

process_domain <- function(prot_domain) {
  pattern_string <- sprintf("^%s.*\\.nwk$", prot_domain)
  emb_files <- list.files(path = paths$emb_folder, pattern = pattern_string, full.names = TRUE)
  attn_files <- list.files(path = paths$attn_folder, pattern = pattern_string, full.names = TRUE)

  njtree_path <- file.path(paths$nj_tree_folder, sprintf("%s.tree", prot_domain))
  mltree_path <- file.path(paths$ml_tree, sprintf("%s.tree", prot_domain))

  nj_tree <- ape::read.tree(njtree_path)
  ml_tree <- ape::read.tree(mltree_path)
  
  calculate_and_write_scores <- function(tree_files, type) {
    for (tree_file in tree_files) {
      current_tree <- ape::read.tree(tree_file)

      nj_rf <- RobinsonFoulds(current_tree, 
                              nj_tree, 
                              similarity = TRUE, 
                              normalize = TRUE)
      ml_rf <- RobinsonFoulds(current_tree, 
                              ml_tree, 
                              similarity = TRUE, 
                              normalize = TRUE)
      nj_ci <- MutualClusteringInfo(current_tree, 
                                    nj_tree, 
                                    normalize = TRUE)
      ml_ci <- MutualClusteringInfo(current_tree, 
                                    ml_tree, 
                                    normalize = TRUE)

      base_name <- basename(sub("\\.nwk$", "", tree_file))
      data <- data.frame(FileName = base_name,
                         NJRFScore = nj_rf,
                         MLRFScore = ml_rf,
                         NJCI = nj_ci,
                         MLCI = ml_ci)
      file_path <- switch(type, 
                          emb = paths$emb_file, 
                          attn = paths$attn_file)
      write_scores(data, file_path)
    }
  }
  
  calculate_and_write_scores(emb_files, "emb")
  calculate_and_write_scores(attn_files, "attn")
  nj_ml_rf_score <- RobinsonFoulds(nj_tree, 
                                   ml_tree, 
                                   similarity = TRUE, 
                                   normalize = TRUE)
  nj_ml_cid <- MutualClusteringInfo(nj_tree, 
                                    ml_tree, 
                                    normalize = TRUE)
  data <- data.frame(ProteinDomain = prot_domain, 
                     RFScore = nj_ml_rf_score, 
                     CI = nj_ml_cid)
  write_scores(data, paths$nj_ml_file)
}

sapply(prot_domains, process_domain)
