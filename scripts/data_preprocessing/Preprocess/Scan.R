start_time = Sys.time()
library(optparse)

option_list = list(
  make_option(c("-f", "--datafile"), type="character", default=NULL,  metavar="character"),
  make_option(c("-v", "--variablelistfile"), type="character", default=NULL, metavar="character"),
  make_option(c("-c", "--datacodingfile"), type="character", default=NULL, metavar="character"),
  make_option(c("-p", "--prsDir"), type="character", default=NULL, metavar="character"),
  make_option(c("-i", "--MRIlistfile"), type="character", default=NULL, metavar="character"),
  make_option(c("-m", "--missingParam"), type="character", default=NULL, metavar="character"),
  make_option(c("-r", "--resultDir"), type="character", default=NULL, metavar="character")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

source("LoadData.R")
source("Reassignment.R")
source("Createout.R")
source("ReplaceNaN.R")
source("Makedataframe.R")
source("Continuous.R")
source("Integer.R")
source("Categorical.R")
source("equalSizedBins.R")
source("TrainValTestsplit.R")
source("GenerateIndex.R")
source("ExtractFields.R")
source("RemovePilot.R")
source("ManualMissing.R")
source("MRI_Select.R")

resultDir = normalizePath(opt$resultDir)
logDir = file.path(resultDir, "log.txt")
discardDir=file.path(resultDir, "discard.RData")
centerDir = file.path(resultDir, "center.csv")



  sink(logDir,append=FALSE,split = FALSE)
  discardlist = list()
  cat("-------------------------Start logging------------------------\n")
  cat("-------------------------Loading Data-------------------------\n")
  LoadData(opt$datafile,opt$variablelistfile, opt$datacodingfile, opt$prsDir, opt$missingParam)
  write.csv(pheno$eid, file.path(resultDir, "eid.csv"), row.names=FALSE)
  cat("---------------------------Finished---------------------------\n")
  cat("----------------------Extracting Fields-----------------------\n")
  Center(pheno, centerDir)
  Extract_fields(pheno, showcase, prs)
  write.csv(pheno, file.path(resultDir, "phenofile_origin.csv"), row.names=FALSE)
  cat("---------------------------Finished---------------------------\n")
  cat("-------------------------Remove Pilot-------------------------\n")
  Removepilot(pheno, centerDir, resultDir)
  write.csv(pheno, file.path(resultDir, "phenofile_1.csv"), row.names=FALSE)
  cat("---------------------------Finished---------------------------\n")
  cat("-------------------Manual missing inputation------------------\n")
  Manual_missing(pheno, missinginfo)
  #write.csv(pheno, file.path(resultDir, "phenofile_12.csv"), row.names=FALSE)
  cat("---------------------------Finished---------------------------\n")
  cat("-----------------------Generating index-----------------------\n")
  GenerateIndex(pheno, resultDir)
  LoadIndex(resultDir)
  cat("---------------------------Finished---------------------------\n")
  trainindex <- trainindex + 1
  valindex <- valindex + 1
  testindex <- testindex + 1
  eid <- pheno$eid
  # write.csv(pheno, file.path(resultDir, "phenofile.csv"), row.names=FALSE)
  
  pheno <- pheno[, 2:dim(pheno)[2]]
  cat("Origin file cols count: ", dim(pheno)[2], "\n")
  name = names(pheno)
  fieldname = as.character(sapply(strsplit(name, split = "-"), function(x) return(x[1])))
  trainpheno <- pheno[trainindex,]
  valpheno <- pheno[valindex,]
  testpheno <- pheno[testindex,]
  trainframe <- data.frame(eid = eid[trainindex])
  testframe <- data.frame(eid = eid[testindex])
  valframe <- data.frame(eid = eid[valindex])
  cat("---------------------Processing Variables---------------------\n")
  for (i in 1:dim(showcase)[1]) {
    variable_info = showcase[i,]
    pheno_single = pheno[, fieldname == as.character(variable_info$FieldID)]
    pheno_train = trainpheno[, fieldname == as.character(variable_info$FieldID)]
    pheno_val = valpheno[, fieldname == as.character(variable_info$FieldID)]
    pheno_test = testpheno[, fieldname == as.character(variable_info$FieldID)]
    cat("\n Variable: ", variable_info$FieldID, "|| ")
    if (!is.na(variable_info$Keep)&&variable_info$Keep == "YES") {
      pheno_train = as.data.frame(pheno_train)
      pheno_val = as.data.frame(pheno_val)
      pheno_test = as.data.frame(pheno_test)
      names(pheno_train) = name[fieldname == variable_info$FieldID]
      names(pheno_val) = name[fieldname == variable_info$FieldID]
      names(pheno_test) = name[fieldname == variable_info$FieldID]
      trainframe <- cbind(trainframe, pheno_train)
      valframe <- cbind(valframe, pheno_val)
      testframe <- cbind(testframe, pheno_test)
      cat(" Keep all")
      next
    }
    if (!any(fieldname == as.character(variable_info$FieldID))) {
      cat(" DataField Not Found !")
      next
    }
    pheno_single = Reassignment(pheno_single, variable_info$FieldID)
    if (variable_info$ValueType == "Continuous") {
      after_out = Creatout(pheno_single, variable_info$FieldID)
      if (!is.null(after_out[[2]])) {
        temp <- TrainValTestsplit(after_out[[2]], trainindex, valindex, testindex)
        trainframe <- cbind(trainframe, temp$train)
        valframe <- cbind(valframe, temp$val)
        testframe <- cbind(testframe, temp$test)
      }
      pheno_single = after_out[[1]]
      temp <- xContinuous(pheno_single, variable_info$FieldID, trainindex, valindex, testindex)
      if(!is.null(temp)){
        trainframe <- cbind(trainframe, temp$train)
        valframe <- cbind(valframe, temp$val)
        testframe <- cbind(testframe, temp$test)
      }
    } else if (variable_info$ValueType == "Integer") {
      after_out = Creatout(pheno_single, variable_info$FieldID)
      if (!is.null(after_out[[2]])) {
        temp <- TrainValTestsplit(after_out[[2]], trainindex, valindex, testindex)
        trainframe <- cbind(trainframe, temp$train)
        valframe <- cbind(valframe, temp$val)
        testframe <- cbind(testframe, temp$test)
      }
      pheno_single = after_out[[1]]
      temp <- xInteger(pheno_single, variable_info$FieldID, trainindex, valindex, testindex)
      if(!is.null(temp)){
        trainframe <- cbind(trainframe, temp$train)
        valframe <- cbind(valframe, temp$val)
        testframe <- cbind(testframe, temp$test)
      }
    } else{
      after_out = Creatout(pheno_single, variable_info$FieldID)
      if (!is.null(after_out[[2]])) {
        temp <- TrainValTestsplit(after_out[[2]], trainindex, valindex, testindex)
        trainframe <- cbind(trainframe, temp$train)
        valframe <- cbind(valframe, temp$val)
        testframe <- cbind(testframe, temp$test)
      }
      pheno_single = after_out[[1]]
      if (!is.null(pheno_single)) {
        temp <- xCategorical(pheno_single, variable_info$FieldID,trainindex, valindex, testindex, discardlist)
        trainframe <- cbind(trainframe, temp$train)
        valframe <- cbind(valframe, temp$val)
        testframe <- cbind(testframe, temp$test)
      }
    }
  }
  cat("\n-----------------Finished Processing Variables----------------\n")
  cat("--------------------------Merge PRS---------------------------\n")
  PRSindex <- grep("PRSPC",names(pheno))
  if(length(PRSindex) > 0){
    trainframe <- cbind(trainframe, trainpheno[,PRSindex])
    valframe <- cbind(valframe, valpheno[,PRSindex])
    testframe <- cbind(testframe, testpheno[,PRSindex])
    cat("PRS Principle Components: ", length(PRSindex) , sep = "")
  }
  cat("\n---------------------------Finished---------------------------\n")
  cat("------------------------Write Results-------------------------\n")
  #write.csv(trainframe, file.path(resultDir, "train.csv"), row.names = FALSE)
  #write.csv(valframe, file.path(resultDir, "val.csv"), row.names = FALSE)
  #write.csv(testframe, file.path(resultDir, "test.csv"), row.names = FALSE)
  finalframe <- data.frame(matrix(nrow = nrow(pheno), ncol = ncol(trainframe)))
  finalframe[trainindex,] <- trainframe
  finalframe[valindex,] <- valframe
  finalframe[testindex,] <- testframe
  names(finalframe) <- names(trainframe)
  write.csv(finalframe, file.path(resultDir, "phenofile_final.csv"), row.names = FALSE)
  save(list = c("discardlist"),file = discardDir)
  cat("---------------------------Finished---------------------------\n")
  cat("------------------------Processing MRI------------------------\n")
  if(!is.null(opt$MRIlistfile)){
    MRI_Select(opt$datafile, opt$MRIlistfile, file.path(resultDir, "eid_remove.rda"), resultDir)
  }
  cat("---------------------Finished Processing MRI------------------\n")
  end_time = Sys.time()
  cat("Time taken:", as.integer(end_time)-as.integer(start_time),"s", "\n")
  cat("--------------------------End logging-------------------------\n")
  sink()
  


