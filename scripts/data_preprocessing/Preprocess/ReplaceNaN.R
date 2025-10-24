ReplaceNaN <- function(pheno) {
    phenoReplaced = pheno
    nanx  = which(is.nan(phenoReplaced))
    phenoReplaced[nanx] = NA;
    
    emptyStr  = which(phenoReplaced=="")	
    phenoReplaced[emptyStr] = NA;
    return(phenoReplaced)
}
